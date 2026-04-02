#!/usr/bin/env python3
"""
Generate a professional banner for ContextBudget project
"""

from PIL import Image, ImageDraw, ImageFont
import os

# Configuration
WIDTH = 1200
HEIGHT = 400
OUTPUT_PATH = "/home/wy517954/code/ContextBudget/pic/banner.png"

# Color scheme (modern gradient)
COLORS = {
    'bg_start': (30, 41, 59),      # Dark slate
    'bg_end': (15, 23, 42),        # Darker slate
    'accent': (59, 130, 246),      # Blue
    'accent_light': (96, 165, 250), # Light blue
    'text_primary': (248, 250, 252), # White
    'text_secondary': (148, 163, 184), # Gray
    'highlight': (251, 191, 36),   # Amber
}

def create_gradient_background(width, height):
    """Create a vertical gradient background"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)

    # Create gradient
    for y in range(height):
        ratio = y / height
        r = int(COLORS['bg_start'][0] * (1 - ratio) + COLORS['bg_end'][0] * ratio)
        g = int(COLORS['bg_start'][1] * (1 - ratio) + COLORS['bg_end'][1] * ratio)
        b = int(COLORS['bg_start'][2] * (1 - ratio) + COLORS['bg_end'][2] * ratio)
        draw.rectangle([(0, y), (width, y + 1)], fill=(r, g, b))

    # Add subtle pattern
    for i in range(0, width + height, 40):
        draw.line([(i, 0), (0, i)], fill=COLORS['accent'], width=1)

    return img

def draw_text_with_shadow(draw, text, font, x, y, color, shadow_color=(0, 0, 0), shadow_offset=2):
    """Draw text with shadow effect"""
    # Draw shadow
    draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=shadow_color)
    # Draw main text
    draw.text((x, y), text, font=font, fill=color)

def main():
    # Create background
    img = create_gradient_background(WIDTH, HEIGHT)
    draw = ImageDraw.Draw(img)

    # Try to load fonts, fall back to default if not available
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        stat_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        stat_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Draw decorative elements
    # Top accent line
    draw.rectangle([(0, 0), (WIDTH, 8)], fill=COLORS['accent'])

    # Bottom accent line
    draw.rectangle([(0, HEIGHT - 8), (WIDTH, HEIGHT)], fill=COLORS['accent'])

    # Side accent bars
    draw.rectangle([(0, 0), (8, HEIGHT)], fill=COLORS['accent'])
    draw.rectangle([(WIDTH - 8, 0), (WIDTH, HEIGHT)], fill=COLORS['accent'])

    # Draw title
    title = "ContextBudget"
    title_width = draw.textlength(title, font=title_font)
    title_x = (WIDTH - title_width) // 2
    draw_text_with_shadow(draw, title, title_font, title_x, 80, COLORS['text_primary'])

    # Draw subtitle
    subtitle = "Budget-Aware Context Management for Long-Horizon Search Agents"
    subtitle_width = draw.textlength(subtitle, font=subtitle_font)
    subtitle_x = (WIDTH - subtitle_width) // 2
    draw.text((subtitle_x, 150), subtitle, font=subtitle_font, fill=COLORS['text_secondary'])

    # Draw key stats/metrics
    stats = [
        ("1.6×", "Gains over baselines"),
        ("30B → 235B", "Model efficiency"),
        ("4K-8K", "Context budget"),
    ]

    stat_x_start = 150
    stat_spacing = 350
    stat_y = 230

    for i, (value, label) in enumerate(stats):
        x = stat_x_start + i * stat_spacing

        # Draw value
        value_width = draw.textlength(value, font=stat_font)
        draw_text_with_shadow(draw, value, stat_font, x + (150 - value_width) // 2, stat_y, COLORS['highlight'])

        # Draw label
        label_width = draw.textlength(label, font=small_font)
        draw.text((x + (150 - label_width) // 2, stat_y + 45), label, font=small_font, fill=COLORS['text_secondary'])

    # Draw key features
    features = [
        "🎯 Budget-Aware Formulation",
        "🔄 Budget-Constrained RL",
        "🧠 Adaptive Memory Management",
        "📊 State-of-the-Art Performance",
    ]

    feature_y = 320
    for i, feature in enumerate(features):
        feature_width = draw.textlength(feature, font=small_font)
        feature_x = (WIDTH - feature_width) // 2
        draw.text((feature_x, feature_y + i * 22), feature, font=small_font, fill=COLORS['accent_light'])

    # Save the image
    img.save(OUTPUT_PATH, 'PNG', quality=95)
    print(f"✓ Banner generated successfully: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()