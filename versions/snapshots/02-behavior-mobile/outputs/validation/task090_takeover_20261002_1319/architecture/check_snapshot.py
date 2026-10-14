#!/usr/bin/python3
"""Independent CPU-only checks for a facts-only architecture refresh."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path('/home/user/Robotic/project2')
OUT = ROOT / 'outputs/validation/task090_takeover_20261002_1319/architecture'
BEFORE = OUT / 'source_before'
NS = '{http://www.w3.org/2000/svg}'


def topology(path):
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for name in node.targets:
                if isinstance(name, ast.Name) and name.id in {'NODES', 'EDGES', 'PENDING', 'BIDIRECTIONAL'}:
                    result[name.id] = ast.literal_eval(node.value)
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old = topology(BEFORE / 'draw_architecture.py')
    new = topology(ROOT / 'scripts/draw_architecture.py')
    assert old == new, 'Topology, coordinates or pending/bidirectional routes changed'
    assert len(new['NODES']) == 9 and len(new['EDGES']) == 17
    svg_path = ROOT / 'outputs/architecture/wam_vlm_architecture.svg'
    png_path = ROOT / 'outputs/architecture/wam_vlm_architecture.png'
    svg = ET.parse(svg_path).getroot()
    original = ET.parse(BEFORE / svg_path.name).getroot()
    paths = lambda doc: [(e.tag, sorted(e.attrib.items())) for e in doc.iter() if e.tag in {NS + tag for tag in ('path', 'rect', 'circle', 'linearGradient', 'marker', 'stop')}]
    assert paths(svg) == paths(original), 'Non-text SVG drawing changed'
    text = ' '.join(''.join(e.itertext()) for e in svg.iter(NS + 'text'))
    for stale in ('25K → 30K', 'CPU WORK / NOT WIRED', 'pending producer', 'checks every 5k', '25k → 30k'):
        assert stale not in text, ('Stale label', stale)
    assert '35k' in text.lower(), '35k result omitted'
    assert '40k' in text.lower() and '0/3' in text
    assert '45k' in text.lower() and '50k' in text.lower()
    assert 'rigid_v2' in text.lower()
    assert '2/3' in text and 'full' in text.lower()
    assert 'GT' in text and 'not admitted' in text.lower()
    import gi
    gi.require_version('Rsvg', '2.0')
    from gi.repository import Rsvg
    import cairo
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 3072, 1830)
    viewport = Rsvg.Rectangle()
    viewport.x, viewport.y, viewport.width, viewport.height = 0, 0, 3072, 1830
    handle = Rsvg.Handle.new_from_file(str(svg_path))
    assert handle.render_document(cairo.Context(surface), viewport)
    png = cairo.ImageSurface.create_from_png(str(png_path))
    assert (png.get_width(), png.get_height()) == (3072, 1830)
    assert bytes(png.get_data()) == bytes(surface.get_data()), 'Published local PNG differs from SVG render'
    report = {
        'checked_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Independent CPU architecture text/topology/render check; no simulation or task-success claim',
        'nodes': 9, 'edges': 17,
        'checks': {'ast_topology_identical': True, 'svg_nontext_geometry_identical': True,
                   'stale_labels_absent': True, 'required_fact_labels_present': True,
                   'png_exact_svg_render': True, 'png_size': [3072, 1830]},
        'manual_visual_inspection': 'Must be recorded separately after opening PNG',
        'sha256': {str(path): digest(path) for path in (ROOT / 'scripts/draw_architecture.py', svg_path, png_path)},
        'text_labels': [''.join(e.itertext()) for e in svg.iter(NS + 'text')],
    }
    (OUT / 'independent_render_check.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'text_labels'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
