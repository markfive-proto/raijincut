use clap::{Parser, Subcommand};
use std::fs;
use std::io::BufRead;
use std::path::PathBuf;
use std::process::{self, Command};
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Parser)]
#[command(name = "raijincut", about = "Parallel video processing CLI: thunder-fast editing powered by Rust")]
struct Cli {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Transcribe audio from a video URL using Whisper
    Transcribe {
        #[arg(help = "Video URL to transcribe")]
        url: String,
        #[arg(short, long, help = "Output file path for transcript")]
        output: String,
        #[arg(short, long, help = "Enable speaker diarization")]
        diarize: bool,
    },
    /// Batch transcribe multiple videos from a URL list file
    BatchTranscribe {
        #[arg(help = "File containing one URL per line")]
        url_list: String,
        #[arg(short, long, help = "Output directory")]
        output: String,
        #[arg(short, long, help = "Enable speaker diarization")]
        diarize: bool,
        #[arg(short, long, default_value = "2", help = "Max concurrent jobs")]
        jobs: usize,
    },
    /// Probe a video file for metadata
    Probe {
        #[arg(help = "Input video file")]
        input: String,
    },
    /// Download a video (YouTube, X, Facebook, LinkedIn, ...) as mp4 plus a metadata JSON
    Download {
        #[arg(help = "Video URL to download")]
        url: String,
        #[arg(short, long, default_value = ".", help = "Output directory")]
        output: String,
        #[arg(long, value_name = "BROWSER", help = "Reuse your signed-in browser session, e.g. chrome (needed for most LinkedIn/X/Facebook posts)")]
        cookies_from_browser: Option<String>,
    },
    /// Break a video down: transcript, shots, keyframes, vision notes, audio hits, captions, breakdown.md/json
    Analyze {
        #[arg(help = "Video file, or a URL to download first")]
        input: String,
        #[arg(short, long, default_value = ".", help = "Output directory (results go to <output>/<slug>/)")]
        output: String,
        #[arg(long, env = "RAIJINCUT_WHISPER_MODEL", help = "whisper.cpp ggml model file; without it, mlx_whisper is tried")]
        whisper_model: Option<String>,
        #[arg(long, default_value = "auto", help = "Vision backend: auto (API key, then ollama), api, ollama, claude-cli, none")]
        vision: String,
        #[arg(long, default_value = "0.3", help = "Hard-cut scene score threshold (lower = more cuts)")]
        scene_threshold: f32,
        #[arg(long, value_name = "BROWSER", help = "When input is a URL: reuse your signed-in browser session, e.g. chrome")]
        cookies_from_browser: Option<String>,
    },
    /// Cut a segment from a video (start/end timestamps)
    Cut {
        #[arg(help = "Input video file")]
        input: String,
        #[arg(short, long, help = "Start timestamp (e.g. 00:01:30)")]
        start: String,
        #[arg(short, long, help = "End timestamp (e.g. 00:05:00)")]
        end: String,
        #[arg(short, long, help = "Output video file")]
        output: String,
    },
    /// Crop a video to a target aspect ratio
    Crop {
        #[arg(help = "Input video file")]
        input: String,
        #[arg(short, long, help = "Output video file")]
        output: String,
        #[arg(short, long, help = "Target aspect ratio (e.g. 9:16, 16:9, 1:1, 4:5)")]
        aspect: String,
    },
    /// Burn subtitles into a video
    Subtitle {
        #[arg(help = "Input video file")]
        input: String,
        #[arg(long, help = "SRT subtitle file")]
        srt: String,
        #[arg(short, long, help = "Output video file")]
        output: String,
    },
    /// Convert/re-encode a video with specified quality
    Convert {
        #[arg(help = "Input video file")]
        input: String,
        #[arg(short, long, help = "Output video file")]
        output: String,
        #[arg(short, long, default_value = "23", help = "CRF quality value (lower = better)")]
        crf: u32,
    },
    /// Repurpose a video for a social media platform
    Repurpose {
        #[arg(help = "Video URL to repurpose")]
        url: String,
        #[arg(short, long, help = "Preset name (e.g. tiktok, reels, shorts)")]
        preset: String,
        #[arg(short, long, help = "Clip timestamps as start..end (e.g. 00:00:10..00:00:30)")]
        clips: String,
        #[arg(long, default_value = "false", help = "Burn subtitles into the video")]
        subtitles: bool,
        #[arg(short, long, help = "Output video file")]
        output: String,
    },
    /// Remove filler words from a video using subtitle timing
    RoughCut {
        #[arg(help = "Input video file")]
        input: String,
        #[arg(long, help = "SRT subtitle file")]
        srt: String,
        #[arg(short, long, help = "Output video file")]
        output: String,
    },
    /// List available presets
    Presets,
    /// Full pipeline: download → probe → transcribe+diarize → filler removal → crop → subtitle → effects
    Pipeline {
        #[arg(help = "One or more video URLs", num_args = 1..)]
        urls: Vec<String>,
        #[arg(short, long, default_value = "reels", help = "Preset name (e.g. tiktok, reels, shorts)")]
        preset: String,
        #[arg(short, long, default_value = "./output", help = "Output directory")]
        output: String,
        #[arg(short, long, default_value = "crossfade", help = "Transition effect between segments (crossfade, fade, wipeleft, radial, dissolve, random, etc.)")]
        transition: String,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        Commands::Transcribe { url, output, diarize } => transcribe(&url, &output, diarize),
        Commands::BatchTranscribe { url_list, output, diarize, jobs } => {
            batch_transcribe(&url_list, &output, diarize, jobs)
        }
        Commands::Probe { input } => probe(&input),
        Commands::Download { url, output, cookies_from_browser } => {
            if let Err(e) = try_download(&url, &output, cookies_from_browser.as_deref()) {
                exit_err(&e);
            }
        }
        Commands::Analyze { input, output, whisper_model, vision, scene_threshold, cookies_from_browser } => {
            if let Err(e) = analyze(&input, &output, whisper_model.as_deref(), &vision, scene_threshold, cookies_from_browser.as_deref()) {
                exit_err(&e);
            }
        }
        Commands::Cut { input, start, end, output } => cut(&input, &start, &end, &output),
        Commands::Crop { input, output, aspect } => crop(&input, &output, &aspect),
        Commands::Subtitle { input, srt, output } => subtitle(&input, &srt, &output),
        Commands::Convert { input, output, crf } => convert(&input, &output, crf),
        Commands::Repurpose { url, preset, clips, subtitles, output } => {
            repurpose(&url, &preset, &clips, subtitles, &output)
        }
        Commands::RoughCut { input, srt, output } => rough_cut(&input, &srt, &output),
        Commands::Presets => list_presets(),
        Commands::Pipeline { urls, preset, output, transition } => {
            pipeline(&urls, &preset, &output, &transition)
        }
    }
}

// ============== Helpers ==============

fn run_cmd(cmd: &str, args: &[&str]) -> Result<std::process::Output, String> {
    let output = Command::new(cmd)
        .args(args)
        .output()
        .map_err(|e| format!("Failed to run '{}': {}", cmd, e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "'{}' exited with {}\n{}",
            cmd,
            output.status,
            stderr.trim()
        ));
    }
    Ok(output)
}

/// Run a command and print its stdout if non-empty. Returns the result for error handling.
fn run_cmd_print(cmd: &str, args: &[&str]) -> Result<(), String> {
    let out = run_cmd(cmd, args)?;
    print_stdout(&out);
    Ok(())
}

fn print_stdout(output: &std::process::Output) {
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.is_empty() {
        print!("{}", stdout);
    }
}

fn path_str(p: &std::path::Path) -> String {
    p.to_string_lossy().to_string()
}

fn sanitize_url_name(url: &str, max_len: usize) -> String {
    let name: String = url
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    if name.len() > max_len {
        name[name.len() - max_len..].to_string()
    } else {
        name
    }
}

fn exit_err(msg: &str) -> ! {
    eprintln!("Error: {}", msg);
    process::exit(1);
}

/// Inline Python script to convert Whisper JSON transcript to SRT format.
const JSON_TO_SRT_SCRIPT: &str = r#"
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
with open(sys.argv[2], 'w') as f:
    for i, seg in enumerate(data.get('segments', []), 1):
        s, e = seg['start'], seg['end']
        sh, sm, ss = int(s//3600), int((s%3600)//60), s%60
        eh, em, es = int(e//3600), int((e%3600)//60), e%60
        f.write(f"{i}\n")
        f.write(f"{sh:02d}:{sm:02d}:{ss:06.3f} --> {eh:02d}:{em:02d}:{es:06.3f}\n".replace('.', ','))
        f.write(f"{seg['text'].strip()}\n\n")
"#;

// ============== Transcribe ==============

fn find_python_dir() -> PathBuf {
    let relative = PathBuf::from("python");
    if relative.exists() {
        return relative;
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let beside_exe = dir.join("python");
            if beside_exe.exists() {
                return beside_exe;
            }
        }
    }
    let in_repo = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/python"));
    if in_repo.exists() {
        return in_repo;
    }
    relative
}

fn transcribe(url: &str, output: &str, diarize: bool) {
    if let Err(e) = try_transcribe(url, output, diarize) {
        exit_err(&e);
    }
}

fn try_transcribe(url: &str, output: &str, diarize: bool) -> Result<(), String> {
    println!("Transcribing: {}", url);

    let py = find_python_dir();
    let tmp_dir = tempfile::tempdir()
        .map_err(|e| format!("Failed to create temp dir: {}", e))?;
    let tmp = tmp_dir.path();
    let audio_path = tmp.join("audio.%(ext)s");
    let audio_path_str = path_str(&audio_path);

    // Step 1: Download audio
    println!("[1/4] Downloading audio...");
    run_cmd(
        "yt-dlp",
        &["-x", "--audio-format", "mp3", "--audio-quality", "0", "-o", &audio_path_str, url],
    )?;

    // Find the downloaded audio file
    let audio_file = fs::read_dir(tmp)
        .map_err(|e| format!("Failed to read temp dir: {}", e))?
        .filter_map(|e| e.ok())
        .find(|e| e.file_name().to_string_lossy().starts_with("audio."))
        .map(|e| path_str(&e.path()))
        .ok_or_else(|| "Failed to find downloaded audio file".to_string())?;

    // Step 2: Fetch metadata
    println!("[2/4] Fetching metadata...");
    let metadata_json_str = path_str(&tmp.join("metadata.json"));
    let fetch_metadata_script = path_str(&py.join("fetch_metadata.py"));
    run_cmd_print("python3", &[&fetch_metadata_script, url, "-o", &metadata_json_str])?;

    // Step 3: Parallel transcription + diarization
    let transcript_json = tmp.join("transcript.json");
    let transcript_json_str = path_str(&transcript_json);
    let diarization_json = tmp.join("diarization.json");
    let diarization_json_str = path_str(&diarization_json);

    let transcribe_script = path_str(&py.join("transcribe_only.py"));
    let diarize_script = path_str(&py.join("diarize_only.py"));

    if diarize {
        println!("[3/4] Transcribing + diarizing in parallel...");

        let audio_t = audio_file.clone();
        let out_t = transcript_json_str.clone();
        let script_t = transcribe_script.clone();

        let audio_d = audio_file.clone();
        let out_d = diarization_json_str.clone();
        let script_d = diarize_script.clone();

        let t_handle = thread::spawn(move || {
            run_cmd("python3", &[&script_t, &audio_t, "-o", &out_t])
        });
        let d_handle = thread::spawn(move || {
            run_cmd("python3", &[&script_d, &audio_d, "-o", &out_d])
        });

        match t_handle.join() {
            Ok(Ok(out)) => print_stdout(&out),
            Ok(Err(e)) => return Err(format!("Transcription failed: {}", e)),
            Err(_) => return Err("Transcription thread panicked".to_string()),
        }
        match d_handle.join() {
            Ok(Ok(out)) => print_stdout(&out),
            Ok(Err(e)) => {
                eprintln!("Warning: diarization failed ({}), continuing without speaker labels.", e);
            }
            Err(_) => {
                eprintln!("Warning: diarization thread panicked, continuing without speaker labels.");
            }
        }
    } else {
        println!("[3/4] Transcribing...");
        run_cmd_print("python3", &[&transcribe_script, &audio_file, "-o", &transcript_json_str])
            .map_err(|e| format!("Transcription failed: {}", e))?;
    }

    // Step 4: Merge into final Markdown
    println!("[4/4] Merging output...");
    let merge_script = path_str(&py.join("merge_transcript.py"));
    let mut merge_args = vec![
        merge_script.as_str(),
        transcript_json_str.as_str(),
        "-o",
        output,
        "-m",
        metadata_json_str.as_str(),
    ];

    if diarize && diarization_json.exists() {
        merge_args.push("-d");
        merge_args.push(diarization_json_str.as_str());
    }

    run_cmd_print("python3", &merge_args)
        .map_err(|e| format!("Merge failed: {}", e))?;

    println!("Transcript saved to: {}", output);
    Ok(())
}

// ============== Batch Transcribe ==============

fn batch_transcribe(url_list: &str, output_dir: &str, diarize: bool, max_jobs: usize) {
    let file = fs::File::open(url_list)
        .unwrap_or_else(|e| exit_err(&format!("Failed to open URL list '{}': {}", url_list, e)));

    let urls: Vec<String> = std::io::BufReader::new(file)
        .lines()
        .filter_map(|l| l.ok())
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();

    if urls.is_empty() {
        exit_err("No URLs found in the list file");
    }

    let _ = fs::create_dir_all(output_dir);
    let total = urls.len();
    println!("Batch transcribe: {} videos, {} concurrent jobs", total, max_jobs);

    let success_count = Arc::new(Mutex::new(0usize));
    let fail_count = Arc::new(Mutex::new(0usize));

    // Simple thread pool using chunks
    let chunks: Vec<Vec<(usize, String)>> = urls
        .into_iter()
        .enumerate()
        .collect::<Vec<_>>()
        .chunks(max_jobs)
        .map(|c| c.to_vec())
        .collect();

    for batch in chunks {
        let handles: Vec<_> = batch
            .into_iter()
            .map(|(idx, url)| {
                let output_dir = output_dir.to_string();
                let diarize = diarize;
                let total = total;
                let success_count = Arc::clone(&success_count);
                let fail_count = Arc::clone(&fail_count);

                thread::spawn(move || {
                    let safe_name = sanitize_url_name(&url, 60);
                    let output_file = format!("{}/transcript_{}.md", output_dir, safe_name);

                    println!("[{}/{}] Transcribing: {}", idx + 1, total, url);
                    match try_transcribe(&url, &output_file, diarize) {
                        Ok(_) => {
                            *success_count.lock().unwrap() += 1;
                        }
                        Err(e) => {
                            eprintln!("[{}/{}] Failed: {}", idx + 1, total, e);
                            *fail_count.lock().unwrap() += 1;
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            if h.join().is_err() {
                *fail_count.lock().unwrap() += 1;
            }
        }
    }

    let s = *success_count.lock().unwrap();
    let f = *fail_count.lock().unwrap();
    println!("\nBatch complete: {} succeeded, {} failed out of {}", s, f, total);
}

// ============== Probe ==============

fn probe(input: &str) {
    let out = match run_cmd(
        "ffprobe",
        &["-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", input],
    ) {
        Ok(o) => o,
        Err(e) => exit_err(&e),
    };

    let j: serde_json::Value = match serde_json::from_str(&String::from_utf8_lossy(&out.stdout)) {
        Ok(v) => v,
        Err(e) => exit_err(&format!("Failed to parse ffprobe output: {}", e)),
    };

    println!("File: {}", j["format"]["filename"].as_str().unwrap_or("video"));
    println!("Duration: {}s", j["format"]["duration"].as_str().unwrap_or("?"));
    if let Some(streams) = j["streams"].as_array() {
        for s in streams {
            if s["codec_type"] == "video" {
                println!(
                    "Video: {}x{}",
                    s["width"].as_u64().unwrap_or(0),
                    s["height"].as_u64().unwrap_or(0)
                );
            }
            if s["codec_type"] == "audio" {
                println!("Audio: {}", s["codec_name"].as_str().unwrap_or("?"));
            }
        }
    }
}

// ============== Download ==============

/// Fields kept from yt-dlp's info dict. `filepath` is the final merged file.
const META_FIELDS: &str = "id,title,uploader,channel,webpage_url,extractor_key,duration,view_count,like_count,comment_count,upload_date,description,filepath";

/// Downloads `url` into `output` and writes `<video>.json` next to it. Returns the video path.
fn try_download(url: &str, output: &str, cookies_from_browser: Option<&str>) -> Result<PathBuf, String> {
    fs::create_dir_all(output).map_err(|e| format!("Cannot create {}: {}", output, e))?;
    println!("Downloading: {}", url);
    let template = format!("{}/%(title).60B-%(id)s.%(ext)s", output);
    let print = format!("after_move:%(.{{{}}})j", META_FIELDS);
    let mut args = vec![
        "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "--merge-output-format", "mp4", "--restrict-filenames", "--no-playlist",
        "-o", &template, "--no-simulate", "--print", &print,
    ];
    if let Some(browser) = cookies_from_browser {
        args.extend(["--cookies-from-browser", browser]);
    }
    args.push(url);

    let out = Command::new("yt-dlp").args(&args).output().map_err(|e| format!("Failed to run yt-dlp: {}", e))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let lower = err.to_lowercase();
        let login_wall = ["login", "log in", "sign in", "cookies", "authentication", "private", "registered users", "not available"]
            .iter()
            .any(|k| lower.contains(k));
        // logged-out LinkedIn/Facebook/X pages often fail as "unable to extract" rather than "login"
        let social = ["linkedin.com", "facebook.com", "fb.watch", "x.com", "twitter.com", "instagram.com"]
            .iter()
            .any(|h| url.contains(h));
        if login_wall || social {
            let hint = if cookies_from_browser.is_some() {
                "The browser session did not unlock it. Open the post in that browser while signed in, then retry."
            } else {
                "Sign in to the site in Chrome yourself, then rerun with --cookies-from-browser chrome."
            };
            return Err(format!("This post probably needs a logged-in session. {}\n(raijincut never asks for or stores passwords.)\n\n{}", hint, err.trim()));
        }
        return Err(format!("yt-dlp failed:\n{}", err.trim()));
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.lines().rev().find(|l| l.trim_start().starts_with('{')).ok_or("yt-dlp printed no metadata")?;
    let mut meta: serde_json::Value = serde_json::from_str(line).map_err(|e| format!("Bad metadata JSON: {}", e))?;
    let video = PathBuf::from(meta["filepath"].as_str().ok_or("yt-dlp did not report the file path")?);
    meta["source_url"] = serde_json::Value::String(url.to_string());
    let meta_path = video.with_extension("json");
    fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap_or_default())
        .map_err(|e| format!("Cannot write {}: {}", meta_path.display(), e))?;
    println!("Video:    {}", video.display());
    println!("Metadata: {}", meta_path.display());
    Ok(video)
}

// ============== Analyze ==============

fn analyze(
    input: &str,
    output: &str,
    whisper_model: Option<&str>,
    vision: &str,
    scene_threshold: f32,
    cookies_from_browser: Option<&str>,
) -> Result<(), String> {
    let video = if input.starts_with("http://") || input.starts_with("https://") {
        try_download(input, output, cookies_from_browser)?
    } else {
        PathBuf::from(input)
    };
    let script = path_str(&find_python_dir().join("analyze.py"));
    let video_s = path_str(&video);
    let threshold = scene_threshold.to_string();
    let mut args = vec![script.as_str(), video_s.as_str(), "-o", output, "--vision", vision, "--scene-threshold", &threshold];
    if let Some(m) = whisper_model {
        args.extend(["--whisper-model", m]);
    }
    // inherit stdio so progress streams live
    let status = Command::new("python3").args(&args).status().map_err(|e| format!("Failed to run python3: {}", e))?;
    if !status.success() {
        return Err(format!("analyze.py exited with {}", status));
    }
    Ok(())
}

// ============== Cut ==============

fn cut(input: &str, start: &str, end: &str, output: &str) {
    println!("Cut: {} -> {}", start, end);
    match run_cmd("ffmpeg", &["-y", "-i", input, "-ss", start, "-to", end, "-c", "copy", output]) {
        Ok(_) => println!("Saved: {}", output),
        Err(e) => exit_err(&e),
    }
}

// ============== Crop ==============

fn crop(input: &str, output: &str, aspect: &str) {
    println!("Crop to {}...", aspect);

    let p = match run_cmd(
        "ffprobe",
        &[
            "-v", "quiet", "-print_format", "json", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", input,
        ],
    ) {
        Ok(o) => o,
        Err(e) => exit_err(&e),
    };

    let j: serde_json::Value = match serde_json::from_str(&String::from_utf8_lossy(&p.stdout)) {
        Ok(v) => v,
        Err(e) => exit_err(&format!("Failed to parse ffprobe output: {}", e)),
    };

    let w = j["streams"][0]["width"].as_u64().unwrap_or(1920) as f64;
    let h = j["streams"][0]["height"].as_u64().unwrap_or(1080) as f64;

    let (tw, th) = match aspect {
        "9:16" => (9.0, 16.0),
        "16:9" => (16.0, 9.0),
        "1:1" => (1.0, 1.0),
        "4:5" => (4.0, 5.0),
        _ => {
            eprintln!("Warning: unknown aspect ratio '{}', defaulting to 9:16", aspect);
            (9.0, 16.0)
        }
    };

    let (cw, ch) = if w / h > tw / th {
        (h * tw / th, h)
    } else {
        (w, w / (tw / th))
    };
    let (cx, cy) = ((w - cw) / 2.0, (h - ch) / 2.0);
    let filter = format!("crop={}:{}:{}:{}", cw as u32, ch as u32, cx as u32, cy as u32);

    match run_cmd("ffmpeg", &["-y", "-i", input, "-vf", &filter, "-c:a", "copy", output]) {
        Ok(_) => println!("Saved: {}", output),
        Err(e) => exit_err(&e),
    }
}

// ============== Subtitle ==============

fn subtitle(input: &str, srt: &str, output: &str) {
    println!("Burn subtitles...");

    let python_script = r#"
import sys
import moviepy as mp
import pysubs2

input_path = sys.argv[1]
srt_path = sys.argv[2]
output_path = sys.argv[3]

video = mp.VideoFileClip(input_path)
subs = pysubs2.load(srt_path)

def make_subtitle_clip(sub):
    text = sub.text.replace(chr(10), ' ')[:80]
    duration = (sub.end - sub.start) / 1000.0
    start = sub.start / 1000.0
    txt = mp.TextClip(text=text, font="Arial", font_size=36, color="white", stroke_color="black", stroke_width=2, method="caption", size=(int(video.w * 0.9), None))
    txt = txt.with_start(start).with_duration(duration)
    txt = txt.with_position(("center", video.h - 80))
    return txt

subtitle_clips = [make_subtitle_clip(s) for s in subs]
final = mp.CompositeVideoClip([video] + subtitle_clips)
final.write_videofile(output_path, codec='libx264', audio_codec='aac', temp_audiofile="/tmp/temp_audio.m4a", remove_temp=True, logger=None)
print("Done!")
"#;

    match run_cmd("python3", &["-c", python_script, input, srt, output]) {
        Ok(_) => println!("Saved: {}", output),
        Err(e) => exit_err(&e),
    }
}

// ============== Convert ==============

fn convert(input: &str, output: &str, crf: u32) {
    println!("Convert (CRF {})...", crf);
    let crf_str = crf.to_string();
    match run_cmd(
        "ffmpeg",
        &["-y", "-i", input, "-c:v", "libx264", "-crf", &crf_str, "-c:a", "aac", output],
    ) {
        Ok(_) => println!("Saved: {}", output),
        Err(e) => exit_err(&e),
    }
}

// ============== Rough Cut ==============

fn rough_cut(input: &str, srt: &str, output: &str) {
    println!("Rough cut: removing filler words...");

    let python_script = r#"
import sys
import moviepy as mp
import pysubs2

input_path = sys.argv[1]
srt_path = sys.argv[2]
output_path = sys.argv[3]

video = mp.VideoFileClip(input_path)
subs = pysubs2.load(srt_path)

fillers = {'um', 'uh', 'er', 'ah', 'like', 'basically', 'actually', 'literally', 'honestly', 'okay', 'right', 'i mean'}
kept = [s for s in subs if not all(w.lower() in fillers for w in s.text.split())]
removed = [s for s in subs if all(w.lower() in fillers for w in s.text.split())]
print(f"Kept {len(kept)} / {len(subs)} subs, removing {len(removed)} filler segments")

# Build list of time ranges to keep (non-filler segments)
keep_ranges = []
for s in kept:
    start = s.start / 1000.0
    end = s.end / 1000.0
    keep_ranges.append((start, end))

# Merge overlapping/adjacent ranges
keep_ranges.sort()
merged = []
for start, end in keep_ranges:
    if merged and start <= merged[-1][1] + 0.1:
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    else:
        merged.append((start, end))

# Extract and concatenate kept segments
clips = [video.subclipped(max(0, s - 0.05), min(video.duration, e + 0.05)) for s, e in merged]
if clips:
    final = mp.concatenate_videoclips(clips)
else:
    final = video

final.write_videofile(output_path, codec='libx264', audio_codec='aac', temp_audiofile="/tmp/temp_audio.m4a", remove_temp=True, logger=None)
print("Done!")
"#;

    match run_cmd("python3", &["-c", python_script, input, srt, output]) {
        Ok(_) => println!("Rough cut done: {}", output),
        Err(e) => exit_err(&e),
    }
}

// ============== Presets ==============

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct PresetVideo {
    width: Option<u32>,
    height: Option<u32>,
    aspect_ratio: Option<String>,
    max_duration: Option<u32>,
    codec: Option<String>,
    bitrate: Option<String>,
    fps: Option<u32>,
}

#[derive(serde::Deserialize, Clone)]
#[allow(dead_code)]
struct PresetSubtitle {
    font: Option<String>,
    font_size: Option<u32>,
    font_color: Option<String>,
    position: Option<String>,
    offset_y: Option<u32>,
    outline_color: Option<String>,
    outline_width: Option<u32>,
    background_color: Option<String>,
    background_padding: Option<u32>,
}

#[derive(serde::Deserialize)]
struct PresetFile {
    video: Option<PresetVideo>,
    subtitle: Option<PresetSubtitle>,
}

fn load_preset(path: &std::path::Path) -> Option<PresetFile> {
    let contents = fs::read_to_string(path).ok()?;
    toml::from_str(&contents).ok()
}

fn find_presets_dir() -> std::path::PathBuf {
    // Try relative path first (running from project root)
    let relative = std::path::PathBuf::from("presets");
    if relative.exists() {
        return relative;
    }
    // Try next to the executable (installed binary)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let beside_exe = dir.join("presets");
            if beside_exe.exists() {
                return beside_exe;
            }
        }
    }
    let in_repo = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/presets"));
    if in_repo.exists() {
        return in_repo;
    }
    relative
}

fn list_presets() {
    let presets_dir = find_presets_dir();
    if !presets_dir.exists() {
        eprintln!("No presets directory found.");
        process::exit(1);
    }

    let mut entries: Vec<_> = match fs::read_dir(presets_dir) {
        Ok(rd) => rd.filter_map(|e| e.ok()).collect(),
        Err(e) => exit_err(&format!("Failed to read presets directory: {}", e)),
    };
    entries.sort_by_key(|e| e.file_name());

    println!("{:<15} {:<12} {:<10} {:<8} {:<8} {:<10}", "PRESET", "RESOLUTION", "ASPECT", "FPS", "MAX(s)", "CODEC");
    println!("{}", "-".repeat(65));

    for entry in &entries {
        let path = entry.path();
        if path.extension().map_or(true, |ext| ext != "toml") {
            continue;
        }
        let name = path.file_stem().unwrap_or_default().to_string_lossy();
        if name == "default" {
            continue;
        }

        if let Some(preset) = load_preset(&path) {
            if let Some(v) = &preset.video {
                let res = match (v.width, v.height) {
                    (Some(w), Some(h)) => format!("{}x{}", w, h),
                    _ => "-".to_string(),
                };
                println!(
                    "{:<15} {:<12} {:<10} {:<8} {:<8} {:<10}",
                    name,
                    res,
                    v.aspect_ratio.as_deref().unwrap_or("-"),
                    v.fps.map_or("-".to_string(), |f| f.to_string()),
                    v.max_duration.map_or("-".to_string(), |d| d.to_string()),
                    v.codec.as_deref().unwrap_or("-"),
                );
            } else {
                println!("{:<15} (no video settings)", name);
            }
        }
    }
}

// ============== Repurpose ==============

fn repurpose(url: &str, preset: &str, clips: &str, subtitles: bool, output: &str) {
    println!("Repurpose: {} for {}", url, preset);

    // Load preset
    let preset_path = find_presets_dir().join(format!("{}.toml", preset));
    let preset_config = match load_preset(&preset_path) {
        Some(p) => p,
        None => exit_err(&format!(
            "Preset '{}' not found. Run 'raijincut presets' to see available presets.",
            preset
        )),
    };

    let aspect = preset_config
        .video
        .as_ref()
        .and_then(|v| v.aspect_ratio.clone())
        .unwrap_or_else(|| "9:16".to_string());

    // Step 1: Download
    let tmp_dir_guard = tempfile::tempdir().unwrap_or_else(|e| {
        exit_err(&format!("Failed to create temp dir: {}", e));
    });
    let tmp_dir = tmp_dir_guard.path().to_string_lossy().to_string();
    let downloaded = format!("{}/downloaded.mp4", tmp_dir);
    println!("[1/4] Downloading...");
    match run_cmd("yt-dlp", &[
        "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "-o", &downloaded, "--force-overwrites", "--merge-output-format", "mp4", url,
    ]) {
        Ok(_) => {}
        Err(e) => exit_err(&e),
    }

    // Step 2: Cut clips
    let cut_file = format!("{}/cut.mp4", tmp_dir);
    let parts: Vec<&str> = clips.splitn(2, "..").collect();
    if parts.len() == 2 {
        println!("[2/4] Cutting: {} -> {}", parts[0], parts[1]);
        match run_cmd(
            "ffmpeg",
            &["-y", "-i", &downloaded, "-ss", parts[0], "-to", parts[1], "-c", "copy", &cut_file],
        ) {
            Ok(_) => {}
            Err(e) => exit_err(&e),
        }
    } else {
        println!("[2/4] No clip range, using full video.");
        if let Err(e) = fs::copy(&downloaded, &cut_file) {
            exit_err(&format!("Failed to copy file: {}", e));
        }
    }

    // Step 3: Crop to preset aspect ratio and scale to target resolution
    let cropped_file = format!("{}/cropped.mp4", tmp_dir);
    println!("[3/4] Cropping to {}...", aspect);
    crop(&cut_file, &cropped_file, &aspect);

    // Scale to preset resolution if specified
    if let Some(ref video) = preset_config.video {
        if let (Some(tw), Some(th)) = (video.width, video.height) {
            let scaled_file = format!("{}/scaled.mp4", tmp_dir);
            let scale_filter = format!("scale={}:{}", tw, th);
            println!("     Scaling to {}x{}...", tw, th);
            match run_cmd(
                "ffmpeg",
                &["-y", "-i", &cropped_file, "-vf", &scale_filter, "-c:a", "copy", &scaled_file],
            ) {
                Ok(_) => {
                    let _ = fs::rename(&scaled_file, &cropped_file);
                }
                Err(e) => eprintln!("Warning: scaling failed ({}), using cropped output.", e),
            }
        }
    }

    // Step 4: Subtitles (optional)
    if subtitles {
        println!("[4/4] Transcribing and burning subtitles...");
        let py = find_python_dir();

        // Extract audio from cropped video
        let audio_file = format!("{}/audio.mp3", tmp_dir);
        match run_cmd(
            "ffmpeg",
            &["-y", "-i", &cropped_file, "-vn", "-acodec", "libmp3lame", "-q:a", "0", &audio_file],
        ) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Warning: audio extraction failed ({}), saving without subtitles.", e);
                if let Err(e) = fs::copy(&cropped_file, output) {
                    exit_err(&format!("Failed to copy file: {}", e));
                }
                println!("Repurpose complete: {}", output);
                return;
            }
        }

        // Transcribe to JSON
        let transcript_json = format!("{}/transcript.json", tmp_dir);
        let transcribe_script = path_str(&py.join("transcribe_only.py"));
        match run_cmd("python3", &[&transcribe_script, &audio_file, "-o", &transcript_json]) {
            Ok(out) => print_stdout(&out),
            Err(e) => {
                eprintln!("Warning: transcription failed ({}), saving without subtitles.", e);
                if let Err(e) = fs::copy(&cropped_file, output) {
                    exit_err(&format!("Failed to copy file: {}", e));
                }
                println!("Repurpose complete: {}", output);
                return;
            }
        }

        // Convert JSON to SRT
        let srt_file = format!("{}/subs.srt", tmp_dir);
        match run_cmd("python3", &["-c", JSON_TO_SRT_SCRIPT, &transcript_json, &srt_file]) {
            Ok(_) => {
                subtitle(&cropped_file, &srt_file, output);
            }
            Err(e) => {
                eprintln!("Warning: SRT generation failed ({}), saving without subtitles.", e);
                if let Err(e) = fs::copy(&cropped_file, output) {
                    exit_err(&format!("Failed to copy file: {}", e));
                }
            }
        }
    } else {
        println!("[4/4] Skipping subtitles.");
        if let Err(e) = fs::copy(&cropped_file, output) {
            exit_err(&format!("Failed to copy file: {}", e));
        }
    }

    println!("Repurpose complete: {}", output);
}

// ============== Styled Subtitle (uses preset config) ==============

fn subtitle_styled(input: &str, srt: &str, output: &str, sub_config: Option<&PresetSubtitle>) {
    println!("Burn subtitles (styled)...");

    let font = sub_config.and_then(|s| s.font.as_deref()).unwrap_or("Arial");
    let font_size = sub_config.and_then(|s| s.font_size).unwrap_or(48);
    let font_color = sub_config
        .and_then(|s| s.font_color.as_deref())
        .unwrap_or("#FFFFFF")
        .to_string();
    let font_color = if font_color.starts_with('#') { font_color } else { format!("#{}", font_color) };
    let stroke_color = sub_config
        .and_then(|s| s.outline_color.as_deref())
        .unwrap_or("#000000")
        .to_string();
    let stroke_color = if stroke_color.starts_with('#') { stroke_color } else { format!("#{}", stroke_color) };
    let stroke_width = sub_config.and_then(|s| s.outline_width).unwrap_or(2);
    let offset_y = sub_config.and_then(|s| s.offset_y).unwrap_or(100);

    let python_script = [
        "import sys",
        "import moviepy as mp",
        "import pysubs2",
        "",
        "input_path = sys.argv[1]",
        "srt_path = sys.argv[2]",
        "output_path = sys.argv[3]",
        "",
        "video = mp.VideoFileClip(input_path)",
        "subs = pysubs2.load(srt_path)",
        "",
        &format!("FONT = \"{}\"", font),
        &format!("FONT_SIZE = {}", font_size),
        &format!("FONT_COLOR = \"{}\"", font_color),
        &format!("STROKE_COLOR = \"{}\"", stroke_color),
        &format!("STROKE_WIDTH = {}", stroke_width),
        &format!("OFFSET_Y = {}", offset_y),
        "",
        "def make_subtitle_clip(sub):",
        "    text = sub.text.replace(chr(10), ' ')[:120]",
        "    if not text.strip():",
        "        return None",
        "    duration = (sub.end - sub.start) / 1000.0",
        "    start = sub.start / 1000.0",
        "    if duration <= 0 or start < 0:",
        "        return None",
        "    txt = mp.TextClip(",
        "        text=text, font=FONT, font_size=FONT_SIZE,",
        "        color=FONT_COLOR, stroke_color=STROKE_COLOR,",
        "        stroke_width=STROKE_WIDTH, method='caption',",
        "        size=(int(video.w * 0.85), None)",
        "    )",
        "    txt = txt.with_start(start).with_duration(duration)",
        "    txt = txt.with_position(lambda t, clip=txt: ('center', video.h - OFFSET_Y - clip.h))",
        "    return txt",
        "",
        "subtitle_clips = [make_subtitle_clip(s) for s in subs]",
        "subtitle_clips = [c for c in subtitle_clips if c is not None]",
        "final = mp.CompositeVideoClip([video] + subtitle_clips)",
        "final.write_videofile(",
        "    output_path, codec='libx264', audio_codec='aac',",
        "    temp_audiofile='/tmp/temp_audio_sub.m4a', remove_temp=True, logger=None",
        ")",
        "print('Done!')",
    ].join("\n");

    match run_cmd("python3", &["-c", &python_script, input, srt, output]) {
        Ok(_) => println!("Saved: {}", output),
        Err(e) => exit_err(&e),
    }
}

// ============== Pipeline ==============

fn pipeline(urls: &[String], preset: &str, output_dir: &str, transition: &str) {
    let _ = fs::create_dir_all(output_dir);

    // Load preset
    let preset_path = find_presets_dir().join(format!("{}.toml", preset));
    let preset_config = match load_preset(&preset_path) {
        Some(p) => p,
        None => exit_err(&format!(
            "Preset '{}' not found. Run 'raijincut presets' to see available presets.",
            preset
        )),
    };

    let aspect = preset_config
        .video
        .as_ref()
        .and_then(|v| v.aspect_ratio.clone())
        .unwrap_or_else(|| "9:16".to_string());

    let sub_config = preset_config.subtitle.clone();
    let target_width = preset_config.video.as_ref().and_then(|v| v.width);
    let target_height = preset_config.video.as_ref().and_then(|v| v.height);

    let total = urls.len();
    println!("=== RaijinCut Pipeline ({} video{}) ===", total, if total > 1 { "s" } else { "" });
    println!("Preset: {} | Aspect: {} | Transition: {}", preset, aspect, transition);
    println!();

    if total == 1 {
        if let Err(e) = try_pipeline(
            &urls[0], &aspect, sub_config.as_ref(),
            target_width, target_height, output_dir, transition, 1, 1,
        ) {
            exit_err(&e);
        }
    } else {
        // Process multiple URLs in parallel
        let success = Arc::new(Mutex::new(0usize));
        let failures = Arc::new(Mutex::new(Vec::<String>::new()));

        let handles: Vec<_> = urls
            .iter()
            .enumerate()
            .map(|(idx, url)| {
                let url = url.clone();
                let aspect = aspect.clone();
                let sub_config = sub_config.clone();
                let output_dir = output_dir.to_string();
                let transition = transition.to_string();
                let total = total;
                let success = Arc::clone(&success);
                let failures = Arc::clone(&failures);

                thread::spawn(move || {
                    match try_pipeline(
                        &url, &aspect, sub_config.as_ref(),
                        target_width, target_height, &output_dir, &transition,
                        idx + 1, total,
                    ) {
                        Ok(_) => {
                            *success.lock().unwrap() += 1;
                        }
                        Err(e) => {
                            eprintln!("[{}/{}] Pipeline failed for {}: {}", idx + 1, total, url, e);
                            failures.lock().unwrap().push(format!("{}: {}", url, e));
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            let _ = h.join();
        }

        let s = *success.lock().unwrap();
        let f = failures.lock().unwrap();
        println!("\n=== Pipeline Complete ===");
        println!("{} succeeded, {} failed out of {}", s, f.len(), total);
        if !f.is_empty() {
            println!("\nFailures:");
            for msg in f.iter() {
                println!("  - {}", msg);
            }
        }
    }
}

fn try_pipeline(
    url: &str,
    aspect: &str,
    sub_config: Option<&PresetSubtitle>,
    target_width: Option<u32>,
    target_height: Option<u32>,
    output_dir: &str,
    transition: &str,
    idx: usize,
    total: usize,
) -> Result<(), String> {
    let tag = if total > 1 { format!("[{}/{}] ", idx, total) } else { String::new() };
    let py = find_python_dir();

    let tmp_dir = tempfile::tempdir()
        .map_err(|e| format!("Failed to create temp dir: {}", e))?;
    let tmp = tmp_dir.path();

    // ── Phase 1: Analyze ──

    // ── Step 1: Download video ──
    println!("{}[1/10] Downloading video...", tag);
    let downloaded_str = path_str(&tmp.join("video.mp4"));
    run_cmd("yt-dlp", &[
        "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "-o", &downloaded_str, "--force-overwrites", "--merge-output-format", "mp4", url,
    ])?;

    // ── Step 2: Probe ──
    println!("{}[2/10] Probing video metadata...", tag);
    let probe_out = run_cmd(
        "ffprobe",
        &["-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", &downloaded_str],
    )?;
    let probe_json: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&probe_out.stdout))
            .map_err(|e| format!("Failed to parse probe output: {}", e))?;
    let duration = probe_json["format"]["duration"].as_str().unwrap_or("?");
    if let Some(streams) = probe_json["streams"].as_array() {
        for s in streams {
            if s["codec_type"] == "video" {
                println!(
                    "{}     Video: {}x{}, duration: {}s",
                    tag,
                    s["width"].as_u64().unwrap_or(0),
                    s["height"].as_u64().unwrap_or(0),
                    duration,
                );
            }
        }
    }

    // ── Step 3: Extract audio ──
    println!("{}[3/10] Extracting audio...", tag);
    let audio_file_str = path_str(&tmp.join("audio.mp3"));
    run_cmd("ffmpeg", &[
        "-y", "-i", &downloaded_str, "-vn", "-acodec", "libmp3lame", "-q:a", "0", &audio_file_str,
    ])?;

    // ── Step 4: Transcribe + Diarize in parallel ──
    println!("{}[4/10] Transcribing + diarizing in parallel...", tag);
    let transcript_json = tmp.join("transcript.json");
    let transcript_json_str = path_str(&transcript_json);
    let diarization_json = tmp.join("diarization.json");
    let diarization_json_str = path_str(&diarization_json);

    let transcribe_script = path_str(&py.join("transcribe_only.py"));
    let diarize_script = path_str(&py.join("diarize_only.py"));

    {
        let audio_t = audio_file_str.clone();
        let out_t = transcript_json_str.clone();
        let script_t = transcribe_script.clone();

        let audio_d = audio_file_str.clone();
        let out_d = diarization_json_str.clone();
        let script_d = diarize_script.clone();

        let t_handle = thread::spawn(move || {
            run_cmd("python3", &[&script_t, &audio_t, "-o", &out_t])
        });
        let d_handle = thread::spawn(move || {
            run_cmd("python3", &[&script_d, &audio_d, "-o", &out_d])
        });

        match t_handle.join() {
            Ok(Ok(out)) => print_stdout(&out),
            Ok(Err(e)) => return Err(format!("Transcription failed: {}", e)),
            Err(_) => return Err("Transcription thread panicked".to_string()),
        }
        match d_handle.join() {
            Ok(Ok(out)) => print_stdout(&out),
            Ok(Err(e)) => {
                eprintln!("{}Warning: diarization failed ({}), continuing without speaker labels.", tag, e);
            }
            Err(_) => {
                eprintln!("{}Warning: diarization thread panicked, continuing without speaker labels.", tag);
            }
        }
    }

    // ── Step 4b: Save full transcript (with diarization + metadata) ──
    let metadata_json_str = path_str(&tmp.join("metadata.json"));
    let fetch_metadata_script = path_str(&py.join("fetch_metadata.py"));
    if let Ok(out) = run_cmd("python3", &[&fetch_metadata_script, url, "-o", &metadata_json_str]) {
        print_stdout(&out);
    }

    let safe_name = sanitize_url_name(url, 50);

    let transcript_md = format!("{}/transcript_{}.md", output_dir, safe_name);
    let merge_script = path_str(&py.join("merge_transcript.py"));
    let mut merge_args = vec![
        merge_script.as_str(),
        transcript_json_str.as_str(),
        "-o",
        transcript_md.as_str(),
        "-m",
        metadata_json_str.as_str(),
    ];
    if diarization_json.exists() {
        merge_args.push("-d");
        merge_args.push(diarization_json_str.as_str());
    }
    if let Ok(out) = run_cmd("python3", &merge_args) {
        print_stdout(&out);
    }
    println!("{}     Transcript saved: {}", tag, transcript_md);

    // ── Step 5: AI Highlight Selection ──
    println!("{}[5/10] Selecting highlights via AI...", tag);
    let highlights_json = tmp.join("highlights.json");
    let highlights_json_str = path_str(&highlights_json);
    let highlight_script = path_str(&py.join("highlight_selector.py"));

    run_cmd_print("python3", &[
        &highlight_script, &transcript_json_str, "-o", &highlights_json_str,
    ]).map_err(|e| format!("Highlight selection failed: {}", e))?;

    // Parse highlights
    let highlights_raw = fs::read_to_string(&highlights_json)
        .map_err(|e| format!("Failed to read highlights.json: {}", e))?;
    let highlights: Vec<serde_json::Value> = serde_json::from_str(&highlights_raw)
        .map_err(|e| format!("Failed to parse highlights.json: {}", e))?;

    if highlights.is_empty() {
        return Err("No highlights selected — transcript may be too short or low-quality.".to_string());
    }

    let total_highlight_dur: f64 = highlights.iter().map(|h| {
        let s = h["start"].as_f64().unwrap_or(0.0);
        let e = h["end"].as_f64().unwrap_or(0.0);
        e - s
    }).sum();
    println!("{}     Selected {} highlights totaling {:.1}s", tag, highlights.len(), total_highlight_dur);

    // ── Phase 2: Edit ──

    // ── Step 6: Cut highlight segments from source (stream copy = instant) ──
    println!("{}[6/10] Cutting highlight segments...", tag);
    let clips_dir = tmp.join("clips");
    let _ = fs::create_dir_all(&clips_dir);
    let mut clip_paths: Vec<String> = Vec::new();

    for (i, h) in highlights.iter().enumerate() {
        let start = h["start"].as_f64().unwrap_or(0.0);
        let end = h["end"].as_f64().unwrap_or(0.0);
        let clip_str = path_str(&clips_dir.join(format!("clip_{:03}.mp4", i)));
        let start_str = format!("{:.3}", start);
        let end_str = format!("{:.3}", end);

        match run_cmd("ffmpeg", &[
            "-y", "-ss", &start_str, "-to", &end_str,
            "-i", &downloaded_str, "-c", "copy", "-avoid_negative_ts", "make_zero",
            &clip_str,
        ]) {
            Ok(_) => {
                let dur = end - start;
                println!("{}     Clip {}: {:.1}s - {:.1}s ({:.0}s)", tag, i + 1, start, end, dur);
                clip_paths.push(clip_str);
            }
            Err(e) => {
                eprintln!("{}Warning: failed to cut clip {} ({}), skipping.", tag, i + 1, e);
            }
        }
    }

    if clip_paths.is_empty() {
        return Err("No clips were successfully cut.".to_string());
    }

    // ── Step 7: Remove fillers from each clip (parallel) ──
    println!("{}[7/10] Removing fillers from {} clips...", tag, clip_paths.len());

    let filler_clean_script = format!(
        r#"
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("{py_dir}")), "python"))
sys.path.insert(0, "{py_dir}")
from transcript_filler import detect_fillers, generate_cleaned_srt

srt_path = sys.argv[1]
output_path = sys.argv[2]

fillers = detect_fillers(srt_path)
print(f"Found {{len(fillers)}} filler instances")

if fillers:
    from collections import Counter
    word_counts = Counter(f['word'] for f in fillers)
    for word, count in word_counts.most_common(5):
        print(f"  {{word}}: {{count}}")

generate_cleaned_srt(srt_path, fillers, output_path)
print(f"Cleaned SRT saved to {{output_path}}")
"#,
        py_dir = py.to_string_lossy(),
    );

    let mut cleaned_clip_paths: Vec<String> = Vec::new();
    let mut clip_srt_paths: Vec<String> = Vec::new();

    for (i, clip_path) in clip_paths.iter().enumerate() {
        let clip_audio_str = path_str(&clips_dir.join(format!("clip_{:03}_audio.mp3", i)));
        let clip_transcript_str = path_str(&clips_dir.join(format!("clip_{:03}_transcript.json", i)));
        let clip_srt = clips_dir.join(format!("clip_{:03}.srt", i));
        let clip_srt_str = path_str(&clip_srt);
        let clip_clean_srt = clips_dir.join(format!("clip_{:03}_clean.srt", i));
        let clip_clean_srt_str = path_str(&clip_clean_srt);
        let clip_roughcut_str = path_str(&clips_dir.join(format!("clip_{:03}_cleaned.mp4", i)));

        // Extract audio
        if run_cmd("ffmpeg", &[
            "-y", "-i", clip_path, "-vn", "-acodec", "libmp3lame", "-q:a", "0", &clip_audio_str,
        ]).is_err() {
            cleaned_clip_paths.push(clip_path.clone());
            clip_srt_paths.push(String::new());
            continue;
        }

        // Transcribe clip
        if run_cmd("python3", &[&transcribe_script, &clip_audio_str, "-o", &clip_transcript_str]).is_err() {
            cleaned_clip_paths.push(clip_path.clone());
            clip_srt_paths.push(String::new());
            continue;
        }

        // JSON → SRT
        if run_cmd("python3", &["-c", JSON_TO_SRT_SCRIPT, &clip_transcript_str, &clip_srt_str]).is_err() {
            cleaned_clip_paths.push(clip_path.clone());
            clip_srt_paths.push(String::new());
            continue;
        }

        // Clean fillers from SRT
        match run_cmd("python3", &["-c", &filler_clean_script, &clip_srt_str, &clip_clean_srt_str]) {
            Ok(out) => print_stdout(&out),
            Err(_) => { let _ = fs::copy(&clip_srt, &clip_clean_srt); }
        }

        let use_srt = if clip_clean_srt.exists() { &clip_clean_srt_str } else { &clip_srt_str };

        // Rough-cut: remove filler segments from the clip
        let roughcut_script = r#"
import sys
import moviepy as mp
import pysubs2

input_path = sys.argv[1]
srt_path = sys.argv[2]
output_path = sys.argv[3]

video = mp.VideoFileClip(input_path)
subs = pysubs2.load(srt_path)

keep_ranges = []
for s in subs:
    start = s.start / 1000.0
    end = s.end / 1000.0
    if end > start and s.text.strip():
        keep_ranges.append((start, end))

keep_ranges.sort()
merged = []
for start, end in keep_ranges:
    if merged and start <= merged[-1][1] + 0.15:
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    else:
        merged.append((start, end))

if not merged:
    import shutil
    shutil.copy(input_path, output_path)
else:
    if merged[0][0] > 0.5:
        merged.insert(0, (0, min(0.5, merged[0][0] - 0.1)))
    if merged[-1][1] < video.duration - 0.5:
        merged.append((max(merged[-1][1] + 0.1, video.duration - 0.5), video.duration))

    clips = []
    for s, e in merged:
        cs = max(0, s - 0.05)
        ce = min(video.duration, e + 0.05)
        if ce > cs:
            clips.append(video.subclipped(cs, ce))

    if clips:
        final = mp.concatenate_videoclips(clips)
    else:
        final = video

    final.write_videofile(output_path, codec='libx264', audio_codec='aac', temp_audiofile="/tmp/temp_audio_clip.m4a", remove_temp=True, logger=None)
"#;
        match run_cmd("python3", &["-c", roughcut_script, clip_path, use_srt, &clip_roughcut_str]) {
            Ok(_) => {
                cleaned_clip_paths.push(clip_roughcut_str.clone());
            }
            Err(_) => {
                cleaned_clip_paths.push(clip_path.clone());
            }
        }

        // Save the SRT path for this clip (we'll regenerate after stitching)
        clip_srt_paths.push(use_srt.to_string());

        println!("{}     Clip {} filler removal done.", tag, i + 1);
    }

    // ── Step 8: Stitch clips with transitions ──
    println!("{}[8/10] Stitching {} clips with '{}' transitions...", tag, cleaned_clip_paths.len(), transition);
    let stitched_file = tmp.join("stitched.mp4");
    let stitched_str = path_str(&stitched_file);

    if cleaned_clip_paths.len() == 1 {
        fs::copy(&cleaned_clip_paths[0], &stitched_file)
            .map_err(|e| format!("Failed to copy single clip: {}", e))?;
    } else {
        // Build a Python call that imports and uses create_transitioned_reel
        let clip_list_json = serde_json::to_string(&cleaned_clip_paths)
            .map_err(|e| format!("Failed to serialize clip paths: {}", e))?;

        let stitch_script = r#"
import sys, os, json
sys.path.insert(0, sys.argv[1])
from video_transitions import create_transitioned_reel

clips = json.loads(sys.argv[2])
output = sys.argv[3]
transition = sys.argv[4]

result = create_transitioned_reel(clips, output, transition=transition, transition_duration=0.5)
print(f"Stitched reel: {result}")
"#;

        let py_dir_str = path_str(&py);
        match run_cmd("python3", &["-c", stitch_script, &py_dir_str, &clip_list_json, &stitched_str, transition]) {
            Ok(out) => print_stdout(&out),
            Err(e) => {
                eprintln!("{}Warning: transition stitching failed ({}), using simple concat.", tag, e);
                let concat_file = tmp.join("concat.txt");
                let concat_str = path_str(&concat_file);
                let mut concat_content = String::new();
                for p in &cleaned_clip_paths {
                    concat_content.push_str(&format!("file '{}'\n", p.replace('\'', "'\\''")));
                }
                fs::write(&concat_file, &concat_content)
                    .map_err(|e| format!("Failed to write concat file: {}", e))?;
                run_cmd("ffmpeg", &[
                    "-y", "-f", "concat", "-safe", "0", "-i", &concat_str,
                    "-c", "copy", &stitched_str,
                ])?;
            }
        }
    }

    let video_for_crop = if stitched_file.exists() { &stitched_str } else { &cleaned_clip_paths[0] };

    // ── Step 9: Crop to portrait + scale ──
    println!("{}[9/10] Cropping to {} and scaling...", tag, aspect);
    let cropped_file = tmp.join("cropped.mp4");
    let cropped_str = path_str(&cropped_file);
    crop(video_for_crop, &cropped_str, aspect);

    // Scale to preset resolution
    if let (Some(tw), Some(th)) = (target_width, target_height) {
        let scaled_file = tmp.join("scaled.mp4");
        let scaled_str = path_str(&scaled_file);
        let scale_filter = format!("scale={}:{}", tw, th);
        println!("{}     Scaling to {}x{}...", tag, tw, th);
        match run_cmd(
            "ffmpeg",
            &["-y", "-i", &cropped_str, "-vf", &scale_filter, "-c:a", "copy", &scaled_str],
        ) {
            Ok(_) => {
                let _ = fs::rename(&scaled_file, &cropped_file);
            }
            Err(e) => eprintln!("{}Warning: scaling failed ({}), using cropped output.", tag, e),
        }
    }

    // Re-transcribe the final stitched+cropped video for accurate subtitle timing
    let cropped_audio_str = path_str(&tmp.join("cropped_audio.mp3"));
    run_cmd("ffmpeg", &[
        "-y", "-i", &cropped_str, "-vn", "-acodec", "libmp3lame", "-q:a", "0", &cropped_audio_str,
    ])?;

    let cropped_transcript = tmp.join("cropped_transcript.json");
    let cropped_transcript_str = path_str(&cropped_transcript);
    println!("{}     Re-transcribing final video for accurate subtitles...", tag);
    match run_cmd("python3", &[&transcribe_script, &cropped_audio_str, "-o", &cropped_transcript_str]) {
        Ok(out) => print_stdout(&out),
        Err(e) => {
            eprintln!("{}Warning: re-transcription failed ({}), using original timestamps.", tag, e);
        }
    }

    let final_sub_srt = tmp.join("final_subs.srt");
    let final_sub_srt_str = path_str(&final_sub_srt);
    let sub_source = if cropped_transcript.exists() { &cropped_transcript_str } else { &transcript_json_str };
    run_cmd("python3", &["-c", JSON_TO_SRT_SCRIPT, sub_source, &final_sub_srt_str])?;

    // Clean fillers from the re-transcribed SRT
    let final_clean_srt = tmp.join("final_clean.srt");
    let final_clean_srt_str = path_str(&final_clean_srt);
    if run_cmd("python3", &["-c", &filler_clean_script, &final_sub_srt_str, &final_clean_srt_str]).is_err() {
        let _ = fs::copy(&final_sub_srt, &final_clean_srt);
    }
    let burn_srt = if final_clean_srt.exists() { &final_clean_srt_str } else { &final_sub_srt_str };

    // ── Step 10: Burn subtitles + effects ──
    println!("{}[10/10] Burning subtitles + applying effects...", tag);
    let subtitled_file = tmp.join("subtitled.mp4");
    let subtitled_str = path_str(&subtitled_file);
    subtitle_styled(&cropped_str, burn_srt, &subtitled_str, sub_config);

    // Apply a fade-in/fade-out effect as final polish
    let final_output = format!("{}/reel_{}.mp4", output_dir, safe_name);
    let fade_duration = "0.5";

    // Get video duration for fade-out offset
    let dur_out = run_cmd("ffprobe", &[
        "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", &subtitled_str,
    ]);
    let video_duration: f64 = dur_out
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse().ok())
        .unwrap_or(30.0);
    let fade_out_start = format!("{:.2}", (video_duration - 0.5).max(0.0));

    let fade_filter = format!(
        "fade=t=in:st=0:d={fd},fade=t=out:st={fo}:d={fd}",
        fd = fade_duration,
        fo = fade_out_start,
    );
    let afade_filter = format!(
        "afade=t=in:st=0:d={fd},afade=t=out:st={fo}:d={fd}",
        fd = fade_duration,
        fo = fade_out_start,
    );

    match run_cmd("ffmpeg", &[
        "-y", "-i", &subtitled_str,
        "-vf", &fade_filter,
        "-af", &afade_filter,
        "-c:v", "libx264", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        &final_output,
    ]) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("{}Warning: fade effect failed ({}), copying without effects.", tag, e);
            fs::copy(&subtitled_file, &final_output)
                .map_err(|e| format!("Failed to copy output: {}", e))?;
        }
    }

    println!("\n{}=== Pipeline Complete ===", tag);
    println!("{}Transcript: {}", tag, transcript_md);
    println!("{}Reel:       {}", tag, final_output);
    println!("{}Duration:   {:.1}s (from {:.1}s highlights)", tag, video_duration, total_highlight_dur);

    Ok(())
}
