use std::cmp::Ordering;
use std::error::Error;
use std::ffi::OsString;
use std::fmt::{Display, Formatter, self};
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::{fs, path::PathBuf, ffi::OsStr};

use clap::Parser;
use image::{ImageBuffer, Rgba, ImageError, ImageOutputFormat, Pixel};
use image::io::Reader as ImageReader;
use os_str_bytes::{Pattern, RawOsStr};
use walkdir::WalkDir;

type ImageBuf = ImageBuffer<Rgba<u8>, Vec<u8>>;

struct ToProcess {
	shape: PathBuf,
	overlay: Option<PathBuf>,
	output: PathBuf
}

impl ToProcess {
	fn process(&self, background: &ImageBuf, algorithm: Algorithm) -> Result<ImageBuf, ProcessingError> {
		let shape_img = ImageReader::open(&self.shape)?.decode()?.into_rgba8();

		let overlay = if let Some(overlay_path) = &self.overlay {
			Some(ImageReader::open(overlay_path)?.decode()?.into_rgba8())
		} else {
			None
		};

		let mut shape_rect = Rect {w: shape_img.width(), h: shape_img.height()};
		let mut bg_rect = Rect {w: background.width(), h: background.height()};

		let (mut shape_fac, mut bg_fac) = calc_factors(&shape_rect, &bg_rect,
			|r1, r2| ProcessingError::IncompatibleBGSize(r1, r2))?;
		
		// keep this rect up to date, we need it for the final buf's dimensions
		bg_rect.scale(bg_fac);
		// and this one for the overlay calcs
		shape_rect.scale(shape_fac);

		// overlay stuff
		let overlay_sampler = if let Some(overlay) = &overlay {
			let mut overlay_rect = Rect {w: overlay.width(), h: overlay.height()};

			let (main_fac, overlay_fac) = calc_factors(&shape_rect, &overlay_rect,
				|r1, r2| ProcessingError::IncompatibleOverlaySize(r1, r2))?;

			shape_fac *= main_fac;
			bg_fac *= main_fac;

			bg_rect.scale(main_fac);

			overlay_rect.scale(overlay_fac);

			// check for mismatched heights
			if overlay_rect != shape_rect {
				return Err(ProcessingError::IncompatibleOverlaySize(shape_rect, overlay_rect));
			}

			Some(Sampler {
				img: overlay,
				scale: overlay_fac
			})
		} else {
			None
		};

		let shape_sampler = Sampler {
			img: &shape_img,
			scale: shape_fac
		};

		let bg_sampler = Sampler {
			img: background,
			scale: bg_fac
		};

		let sampler = TriSampler {
			shape_sampler, bg_sampler, overlay_sampler, algorithm,
			height_mod: bg_rect.w
		};

		let mut new_bg_buf = ImageBuf::new(bg_rect.w, bg_rect.h);

		for i in 0..bg_rect.w {
			for j in 0..bg_rect.h {
				new_bg_buf.put_pixel(i, j, sampler.get_pixel(i, j));
			}
		}
		
		Ok(new_bg_buf)
	}
}

struct TriSampler<'a, 'b, 'c> {
	shape_sampler: Sampler<'a>,
	bg_sampler: Sampler<'b>,
	overlay_sampler: Option<Sampler<'c>>,
	height_mod: u32,
	algorithm: Algorithm
}

impl TriSampler<'_, '_, '_> {
	fn get_pixel(&self, x: u32, y: u32) -> Rgba<u8> {
		let mod_y = y % self.height_mod;
		if let Some(overlay) = &self.overlay_sampler {
			let overlay_px = overlay.get_pixel(x, mod_y);
			if overlay_px.0[3] > 0 {
				return *overlay_px;
			}
		}

		// blending algo: ((dst * (255 - a) + src * a) + 127) / 255
		// or:  (dst * (255 - a)) / 255 + src
		// dst = bg
		// src = shape
		
		// saturation: lm + (int) (a * (bg - lm))

		let shape_px = self.shape_sampler.get_pixel(x, mod_y);
		if shape_px.0[3] > 0 {
			let bg_px = self.bg_sampler.get_pixel(x, y);
			
			self.algorithm.blend(bg_px, shape_px)
		} else {
			return Rgba([0, 0, 0, 0]);
		}
	}
}



struct Sampler<'a> {
	img: &'a ImageBuf,
	scale: u32
}

impl Sampler<'_> {
	fn get_pixel(&self, x: u32, y: u32) -> &Rgba<u8> {
		self.img.get_pixel(x / self.scale, y / self.scale)
	}
}

fn calc_factors(r1: &Rect, r2: &Rect, err_fn: impl Fn(Rect, Rect) -> ProcessingError) -> Result<(u32, u32), ProcessingError> {
	match r1.w.cmp(&r2.w) {
		Ordering::Less => {
			if r2.w % r1.w != 0 {
				Err(err_fn(*r1, *r2))
			} else {
				Ok((r2.w / r1.w, 1))
			}
		},
		Ordering::Greater => {
			if r1.w % r2.w != 0 {
				Err(err_fn(*r1, *r2))
			} else {
				Ok((1, r1.w / r2.w))
			}
		},
		Ordering::Equal => {
			Ok((1, 1))
		}
	}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rect {
	w: u32,
	h: u32
}

impl Rect {
	fn scale(&mut self, factor: u32) {
		self.w *= factor;
		self.h *= factor;
	}
}

#[derive(Debug)]
enum ProcessingError {
	IOError(io::Error),
	ImageError(ImageError),
	IncompatibleBGSize(Rect, Rect),
	IncompatibleOverlaySize(Rect, Rect)
}

impl From<io::Error> for ProcessingError {
	fn from(from: io::Error) -> Self {
		ProcessingError::IOError(from)
	}
}

impl From<ImageError> for ProcessingError {
    fn from(from: ImageError) -> Self {
        ProcessingError::ImageError(from)
    }
}

impl Display for ProcessingError {
	fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), fmt::Error> {
		match self {
			Self::IOError(ie) => ie.fmt(f),
			Self::ImageError(ie) => ie.fmt(f),
			Self::IncompatibleBGSize(s, b) =>
				write!(f, "Incompatible sizes: Shape({}, {}), BG: ({}, {})", s.w, s.h, b.w, b.h),
			Self::IncompatibleOverlaySize(s, o) =>
				write!(f, "Incompatible sizes: (Scaled) shape({}, {}), overlay: ({}, {})", s.w, s.h, o.w, o.h)
		}
	}
}

impl Error for ProcessingError {
	fn cause(&self) -> Option<&dyn Error> {
		match self {
			Self::IOError(e) => Some(e),
			Self::ImageError(e) => Some(e),
			_ => None
		}
	}
}

#[derive(Parser)]
#[command(version, long_about = None)]
struct Args {
	/// Background
	background: PathBuf,
	/// Frame time (mcmeta)
	#[arg(default_value_t = 1)]
	frametime: u32,
	/// Replacement for $ in input filenames to output
	texture_name: Option<OsString>,
	/// Input directory
	#[arg(short, long, default_value = "shapes")]
	shapes: PathBuf,
	// Output directory
	#[arg(short, long, default_value = "output")]
	output: PathBuf,
	/// Interpolate (mcmeta)
	#[arg(short)]
	interpolate: bool,
	/// Blending algorithm to use
	#[arg(short, long, default_value = "gtnh", value_enum)]
	blend: Algorithm
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum Algorithm {
	Gtnh,
	Multiply
}

impl Algorithm {
	fn blend(&self, bg_px: &Rgba<u8>, shape_px: &Rgba<u8>) -> Rgba<u8> {
		match self {
			Algorithm::Gtnh => {
				let intermediate = Rgba([
					cursed_blend(bg_px.0[0] as u32, shape_px.0[0] as u32, 30),
					cursed_blend(bg_px.0[1] as u32, shape_px.0[1] as u32, 30),
					cursed_blend(bg_px.0[2] as u32, shape_px.0[2] as u32, 30),
					bg_px.0[3]
				]);
				let luma = intermediate.to_luma().0[0] as f32;
				Rgba([
					cursed_saturation(intermediate.0[0] as f32, luma, 1.5),
					cursed_saturation(intermediate.0[1] as f32, luma, 1.5),
					cursed_saturation(intermediate.0[2] as f32, luma, 1.5),
					intermediate.0[3]
				])
			},
			Algorithm::Multiply => {
				Rgba([
					color_mul(bg_px[0], shape_px[1]),
					color_mul(bg_px[1], shape_px[1]),
					color_mul(bg_px[2], shape_px[2]),
					bg_px[3]
				])
			}
		}
	}
}

fn color_mul(c1: u8, c2: u8) -> u8 {
	((c1 as u16 * c2 as u16) / 255) as u8
}

fn cursed_saturation(src: f32, luma: f32, factor: f32) -> u8 {
	let val = luma + (factor * (src - luma));
	if val > 255. {
		255
	} else if val < 0. {
		0
	} else {
		val as u8
	}
}

fn cursed_blend(dst: u32, src: u32, a: u32) -> u8 {
	(((dst * (255 - a) + src * a) + 127) / 255) as u8
}

fn main() {
	let args = Args::parse();

	let output_path = args.output;
	let shapes_path = args.shapes;
	let texture_name = args.texture_name;
	let algorithm = args.blend;

	let mcmeta = format_mcmeta(args.frametime, args.interpolate);

	let background = ImageReader::open(&args.background).expect("Read background").decode().expect("Decode background").into_rgba8();

	let mut files: Vec<ToProcess> = Vec::new();

	for entry in WalkDir::new(&shapes_path) {
		let entry = entry.expect("Dir walk");
		
		if entry.file_type().is_file() {
			let path = entry.path();
			
			if path.extension() == Some(&OsStr::new("png")) && !path.file_stem().expect("File stem").ends_with("_overlay") {
				let stem = path.file_stem().expect("File stem");
				let mut overlay_fname = stem.to_os_string();
				overlay_fname.push("_overlay.png");

				let overlay_path = path.with_file_name(overlay_fname);

				let overlay = if overlay_path.exists() {
					Some(overlay_path)
				} else {
					None
				};

				let mut output = output_path.join(path.strip_prefix(&shapes_path).expect("Path strip prefix"));
				// replace $
				if let Some(name) = &texture_name {
					output = PathBuf::from(output.as_os_str().replace("$", name))
				}

				files.push(ToProcess {
					shape: path.to_owned(),
					overlay, output
				});
			}
		}
	}

	for entry in files {
		if let Err(err) = process_write(&entry, &background, &mcmeta, algorithm) {
			eprintln!("Error processing file {:?}: {}", &entry.shape, err);
		}
	}
}

fn process_write(of: &ToProcess, background: &ImageBuf, mcmeta: &str, algorithm: Algorithm) -> Result<(), ProcessingError> {
	let image = of.process(background, algorithm)?;
	let mut file = create_file_and_parents(&of.output)?;
	image.write_to(&mut file, ImageOutputFormat::Png)?;

	let mcmeta_fname = of.output.with_extension("png.mcmeta");
	let mut mcmeta_file = File::create(mcmeta_fname)?;
	mcmeta_file.write(mcmeta.as_bytes())?;

	Ok(())
}

fn format_mcmeta(frametime: u32, interpolate: bool) -> String {
	format!("{{ \"animation\": {{ \"interpolate\": {}, \"frametime\": {} }} }}", interpolate, frametime)
}

struct Frames(u32);

impl Display for Frames {
	fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), fmt::Error> {
		f.write_str("[0")?;

		for i in 1..self.0 {
			write!(f, ", {}", i)?;
		}

		f.write_str("]")?;

		Ok(())
	}
}

fn create_file_and_parents(path: &Path) -> Result<File, io::Error> {
	if let Some(parent) = path.parent() {
		fs::create_dir_all(parent)?
	}
	File::create(path)
}

trait OsStrHacks {
	fn replace<P: Pattern>(&self, pattern: P, with: &OsStr) -> OsString;
	fn ends_with<P: Pattern>(&self, pattern: P) -> bool;
}

impl OsStrHacks for OsStr {
	fn replace<P: Pattern>(&self, pattern: P, with: &OsStr) -> OsString {
		let raw = RawOsStr::new(self);
		let mut split = raw.split(pattern);

		let mut new_str = OsString::new();
		new_str.push(split.next().expect("Split into no parts?").to_os_str());

		for seg in split {
			new_str.push(with);
			new_str.push(seg.to_os_str());
		}

		new_str
	}

	fn ends_with<P: Pattern>(&self, pattern: P) -> bool {
		RawOsStr::new(self).ends_with(pattern)
	}
}