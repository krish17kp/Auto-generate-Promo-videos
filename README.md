Auto-Generate Promo Videos

Automatically generate promotional videos from text and media assets using AI-driven templates and video editing automation.

Table of Contents

Overview

Key Features

Demo / Screenshots

Installation

Usage

Tech Stack

How It Works

Contributing

License

1. Overview

Auto-Generate Promo Videos is a tool that transforms text and multimedia assets into ready-to-publish promotional videos. It leverages automation and AI to format visual content, apply transitions, and produce cohesive output without manual editing.

This project is designed for content creators, social media managers, and marketers who want to scale promo video creation with minimal effort.

2. Key Features

Generate promo videos from structured input text

Automatic media sequencing and transition application

Template-based video layout and style

Support for images, video clips, and captions

Export videos in standard MP4 format


3. Installation
Requirements

Node.js (v14+)

Python 3.8+

FFmpeg installed and accessible in PATH

Clone Repository
git clone https://github.com/krish17kp/Auto-generate-Promo-videos.git
cd Auto-generate-Promo-videos

Install Dependencies
npm install
pip install -r requirements.txt

4. Usage
Run the Application
npm start

Generate a Video

Prepare a project folder with:

Text script (text.txt)

Media assets (images/videos)

Run the generator:

python generate_video.py --input ./project_folder


Output will be saved to:

./output/video.mp4

5. Tech Stack
Layer	Technology
Frontend	React / HTML / CSS
Backend	Python
Media Processing	FFmpeg
Automation Scripts	Python
6. How It Works

Input Parsing: Reads structured text and asset list.

Template Selection: Applies predefined layout and timing rules.

Media Composition: Uses FFmpeg to merge assets, add transitions and captions.

Export: Generates a final promo video.

7. Contributing

Fork the repository

Create a feature branch

Commit changes with clear messages

Open a pull request

Please follow standard commit and code style guidelines.
