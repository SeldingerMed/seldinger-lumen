"""Build static launch SVG assets used by docs, preprint, and social posts."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape


OUT = Path("docs/assets/launch")


def _text_lines(lines, *, x, y, size, fill, weight=400, line_height=1.18, anchor="start"):
    parts = []
    for i, line in enumerate(lines):
        parts.append(
            f'<text x="{x}" y="{y + i * size * line_height:.1f}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(line)}</text>'
        )
    return "\n".join(parts)


def _shell(width=1200, height=630, *, title="", subtitle="", eyebrow="LUMEN"):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#071012"/>
      <stop offset="0.46" stop-color="#13211f"/>
      <stop offset="1" stop-color="#f4f0e7"/>
    </linearGradient>
    <linearGradient id="cyan" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#52e3d5"/>
      <stop offset="1" stop-color="#f4d35e"/>
    </linearGradient>
    <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="5" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect width="100%" height="100%" fill="url(#bg)"/>
  <rect x="38" y="38" width="{width - 76}" height="{height - 76}" rx="24" fill="none" stroke="#ffffff" stroke-opacity="0.16"/>
  <text x="78" y="96" font-family="Arial, Helvetica, sans-serif" font-size="25" font-weight="700" fill="#f4d35e" letter-spacing="6">{escape(eyebrow)}</text>
  {_text_lines(title.split("|"), x=78, y=178, size=70, fill="#ffffff", weight=800)}
  {_text_lines(subtitle.split("|"), x=82, y=390, size=31, fill="#d8e5df", weight=500)}
'''


def social_card():
    svg = _shell(
        title="The open simulator|for endovascular AI",
        subtitle="Deformable vessel wall. GPU-parallel physics.|Safety-scored RL. CV-ready synthetic fluoro.",
    )
    svg += '''
  <path d="M690 224 C780 220 805 276 880 286 C965 298 1015 245 1110 232" fill="none" stroke="#ef746f" stroke-width="16" stroke-linecap="round" opacity="0.52"/>
  <path d="M690 224 C780 220 805 276 880 286 C965 298 1015 245 1110 232" fill="none" stroke="#ff9b8f" stroke-width="5" stroke-linecap="round"/>
  <path d="M705 230 C776 240 817 302 898 320 C980 338 1030 294 1122 280" fill="none" stroke="#ef746f" stroke-width="13" stroke-linecap="round" opacity="0.45"/>
  <path d="M705 230 C776 240 817 302 898 320 C980 338 1030 294 1122 280" fill="none" stroke="#ff9b8f" stroke-width="4" stroke-linecap="round"/>
  <path d="M642 245 C723 241 790 250 848 288" fill="none" stroke="url(#cyan)" stroke-width="7" stroke-linecap="round" filter="url(#glow)"/>
  <circle cx="848" cy="288" r="11" fill="#f4d35e" stroke="#fff7bd" stroke-width="5"/>
  <circle cx="642" cy="245" r="8" fill="#52e3d5"/>
  <text x="82" y="560" font-family="Arial, Helvetica, sans-serif" font-size="24" fill="#eff7f4">github.com/SeldingerMed/seldinger-lumen</text>
</svg>
'''
    return svg


def architecture_card():
    svg = _shell(
        title="One core, many|intraluminal tasks",
        subtitle="A slender device inside a soft tube: vessels, airways, bowel, ducts.",
        eyebrow="LUMEN ARCHITECTURE",
    )
    labels = [
        ("Procedural anatomy", 690, 170, "#ff9b8f"),
        ("Newton/Warp solver", 820, 280, "#52e3d5"),
        ("Fluoro + labels", 725, 392, "#f4d35e"),
        ("RL benchmark", 900, 500, "#ffffff"),
    ]
    for label, x, y, color in labels:
        svg += f'<rect x="{x}" y="{y}" width="260" height="74" rx="18" fill="#071012" fill-opacity="0.72" stroke="{color}" stroke-opacity="0.75"/>\n'
        svg += f'<text x="{x + 22}" y="{y + 46}" font-family="Arial, Helvetica, sans-serif" font-size="24" font-weight="700" fill="{color}">{escape(label)}</text>\n'
    svg += '''
  <path d="M820 244 L820 280 M850 354 L850 392 M900 466 L900 500" stroke="#ffffff" stroke-opacity="0.35" stroke-width="4" stroke-linecap="round"/>
  <path d="M640 520 C735 470 798 424 850 354 C895 294 972 255 1110 225" fill="none" stroke="url(#cyan)" stroke-width="8" stroke-linecap="round" filter="url(#glow)"/>
  <circle cx="1110" cy="225" r="12" fill="#f4d35e" stroke="#fff7bd" stroke-width="5"/>
</svg>
'''
    return svg


def comparison_card():
    svg = _shell(
        title="Beyond rigid-pipe|catheter tasks",
        subtitle="",
        eyebrow="WHY IT MATTERS",
    )
    features = [
        ("deformable HGO wall", 720, 170),
        ("implicit tube contact", 720, 238),
        ("GPU-parallel Newton/Warp", 720, 306),
        ("fluoro masks + keypoints", 720, 374),
        ("safe success before raw success", 720, 442),
    ]
    for text, x, y in features:
        svg += f'<circle cx="{x - 28}" cy="{y - 8}" r="10" fill="#52e3d5"/>\n'
        svg += f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="25" font-weight="700" fill="#ffffff">{escape(text)}</text>\n'
    svg += _text_lines(
        "Deformable contact, differentiability, safety-scored success,|and CV annotations in one open stack.".split("|"),
        x=82,
        y=500,
        size=29,
        fill="#d8e5df",
        weight=600,
    )
    svg += '</svg>\n'
    return svg


def square_card():
    svg = _shell(
        1080,
        1080,
        title="Lumen",
        subtitle="Open, differentiable, GPU-parallel|simulation for endovascular AI.",
        eyebrow="LAUNCH",
    )
    svg += '''
  <path d="M155 705 C300 690 375 590 520 610 C675 632 720 830 930 790" fill="none" stroke="#ef746f" stroke-width="30" stroke-linecap="round" opacity="0.45"/>
  <path d="M155 705 C300 690 375 590 520 610 C675 632 720 830 930 790" fill="none" stroke="#ff9b8f" stroke-width="8" stroke-linecap="round"/>
  <path d="M145 705 C315 700 410 666 525 626" fill="none" stroke="url(#cyan)" stroke-width="12" stroke-linecap="round" filter="url(#glow)"/>
  <circle cx="525" cy="626" r="20" fill="#f4d35e" stroke="#fff7bd" stroke-width="8"/>
  <text x="82" y="970" font-family="Arial, Helvetica, sans-serif" font-size="32" fill="#eff7f4">seldingermed.github.io/seldinger-lumen</text>
</svg>
'''
    return svg


def video_demo_bg():
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#071012"/>
      <stop offset="1" stop-color="#17231f"/>
    </linearGradient>
  </defs>
  <rect width="1280" height="720" fill="url(#bg)"/>
  <text x="70" y="78" font-family="Arial, Helvetica, sans-serif" font-size="28" font-weight="800" fill="#f4d35e" letter-spacing="5">SAME ROLLOUT</text>
  <text x="366" y="82" font-family="Arial, Helvetica, sans-serif" font-size="40" font-weight="800" fill="#ffffff">control state and clinical image, side by side</text>
  <rect x="68" y="118" width="544" height="544" rx="20" fill="#020606" stroke="#ffffff" stroke-opacity="0.15"/>
  <rect x="668" y="118" width="544" height="544" rx="20" fill="#020606" stroke="#ffffff" stroke-opacity="0.15"/>
  <text x="82" y="696" font-family="Arial, Helvetica, sans-serif" font-size="25" font-weight="800" fill="#ffffff">schematic navigation</text>
  <text x="682" y="696" font-family="Arial, Helvetica, sans-serif" font-size="25" font-weight="800" fill="#ffffff">synthetic fluoroscopy</text>
</svg>
'''


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        "social-card.svg": social_card(),
        "social-card-square.svg": square_card(),
        "architecture-card.svg": architecture_card(),
        "comparison-card.svg": comparison_card(),
        "video-demo-bg.svg": video_demo_bg(),
    }
    for name, data in files.items():
        (OUT / name).write_text(data)
        print(OUT / name)


if __name__ == "__main__":
    main()
