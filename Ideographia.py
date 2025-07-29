import hashlib
import base64
import math
import itertools
import random

import cairo
from io import BytesIO

B64URL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"


def disp_and_save(
    draw_func,
    b64str,
    display=False,
    folder="Icons/",
    filename_png=None,
    filename_svg=None,
    res=256,
):
    width = res
    height = res

    # only supports b64url of up to 16 length, otherwise raises error
    if len(b64str) > 16:
        raise IndexError(f"Maximum string length is 16 but it was {len(b64str)}")

    # Create an ImageSurface for PNG display
    surface_png = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx_png = cairo.Context(surface_png)
    draw_func(ctx_png, b64str, width, height)

    if filename_png:
        surface_png.write_to_png(folder + "PNG/" + filename_png + ".png")

    if filename_svg:
        # Create an SVGSurface to save the image as SVG
        surface_svg = cairo.SVGSurface(
            folder + "SVG/" + filename_svg + ".svg", width, height
        )
        ctx_svg = cairo.Context(surface_svg)
        draw_func(ctx_svg, b64str, width, height)
        surface_svg.finish()

    if display:  # Display using matplotlib
        with BytesIO() as fileobj:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

            surface_png.write_to_png(fileobj)

            fileobj.seek(0)
            img = mpimg.imread(fileobj, format="png")

            plt.imshow(img)
            plt.axis("off")
            plt.show()


def b64url_char_to_int(char):
    if len(char) != 1:
        raise Exception("ERROR: Input was a string of lenght>1, not a character")
    num = ord(char)
    if 65 <= num <= 90:
        num -= 65
    elif 97 <= num <= 122:
        num -= 71
    elif 48 <= num <= 57:
        num += 4
    elif num == 45:
        num += 17
    elif num == 95:
        num -= 32
    else:
        raise ValueError(f"ERROR: '{char}' is not valid in base64url")

    return num


def get_circles():
    circles = []
    for row in range(5):
        rf = row / 4
        for col in range(5):
            cf = col / 4
            if row != 2 or col != 2:
                if (row == 0 or row == 4) and (col == 0 or col == 4):
                    radii = [0.5, 0.75, 1]
                elif row == 0 or row == 4 or col == 0 or col == 4:
                    radii = [0.5, 0.75, 1]
                else:
                    radii = [0.25, 0.5]
                for r in radii:
                    circles.append((rf, cf, r))
    return circles


def get_centers():
    centers = []
    for row in range(9):
        rf = row / 8
        for col in range(9):
            cf = col / 8
            if row != 4 or col != 4:
                if (row == 0 or row == 8) and (
                    col == 0 or col == 1 or col == 4 or col == 7 or col == 8
                ):
                    pass
                elif (col == 0 or col == 8) and (
                    row == 0 or row == 1 or row == 4 or row == 7 or row == 8
                ):
                    pass
                else:
                    centers.append((rf, cf))
    # print(centers)
    # print(len(centers))
    # assert len(centers)==64
    return centers


def get_position(centers, b64str, idx):
    center = centers[b64url_char_to_int(b64str[idx])]
    return center


def set_color(idx, cr: cairo.Context, b64str: str, colors):
    if len(b64str) == (idx + 1):
        cr.set_source_rgb(0, 0, 0)
    else:
        color = colors[b64url_char_to_int(b64str[idx + 1])]
        cr.set_source_rgb(color[0], color[1], color[2])


def draw_circle(cr, circle, idx, b64str, colors, outline_line_width):
    cr.set_source_rgb(0, 0, 0)
    cr.arc(circle[0], circle[1], circle[2], 0, 2 * math.pi)
    cr.fill()
    set_color(idx, cr, b64str, colors)
    cr.arc(circle[0], circle[1], circle[2] - outline_line_width, 0, 2 * math.pi)
    cr.fill()


def trace_square(cr: cairo.Context, l, r, t, b):
    cr.move_to(l, t)
    cr.line_to(r, t)
    cr.line_to(r, b)
    cr.line_to(l, b)
    cr.close_path()


def draw_square(cr, center, outline_line_width):
    sq_side = 0.5
    sq_semiside = sq_side / 2
    sq_line_width = 0.2
    l = center[0] - sq_semiside
    r = center[0] + sq_semiside
    t = center[1] - sq_semiside
    b = center[1] + sq_semiside

    # fill stroke
    cr.set_line_width(sq_line_width)
    trace_square(cr, l, r, t, b)
    cr.stroke()

    # outlines
    cr.set_source_rgb(0, 0, 0)
    cr.set_line_width(outline_line_width)

    sq_line_semiwidth = sq_line_width / 2
    # external outline
    trace_square(
        cr,
        l - sq_line_semiwidth,
        r + sq_line_semiwidth,
        t - sq_line_semiwidth,
        b + sq_line_semiwidth,
    )

    # internal outline
    trace_square(
        cr,
        l + sq_line_semiwidth,
        r - sq_line_semiwidth,
        t + sq_line_semiwidth,
        b - sq_line_semiwidth,
    )

    cr.stroke()


def draw_diamond(cr, center, outline_line_width):
    di_diameter = 0.75
    di_radius = di_diameter / 2
    di_line_width = 0.12
    l = center[0] - di_radius
    r = center[0] + di_radius
    t = center[1] - di_radius
    b = center[1] + di_radius

    # fill stroke
    cr.set_line_width(di_line_width)
    cr.move_to(l, center[1])
    cr.line_to(center[0], t)
    cr.line_to(r, center[1])
    cr.line_to(center[0], b)
    cr.close_path()
    cr.stroke()

    # outlines
    cr.set_source_rgb(0, 0, 0)
    cr.set_line_width(outline_line_width)

    di_aux_dist = di_line_width / math.sqrt(2)

    # external outline
    cr.move_to(l - di_aux_dist, center[1])
    cr.line_to(center[0], t - di_aux_dist)
    cr.line_to(r + di_aux_dist, center[1])
    cr.line_to(center[0], b + di_aux_dist)
    cr.close_path()

    # internal outline
    cr.move_to(l + di_aux_dist, center[1])
    cr.line_to(center[0], t + di_aux_dist)
    cr.line_to(r - di_aux_dist, center[1])
    cr.line_to(center[0], b - di_aux_dist)
    cr.close_path()
    cr.stroke()


def trace_tri(cr, a, b, c):
    cr.move_to(a[0], a[1])
    cr.line_to(b[0], b[1])
    cr.line_to(c[0], c[1])
    cr.close_path()


def draw_uptri(cr, center, outline_line_width):
    tri_diameter = 0.4
    tri_radius = tri_diameter / 2
    tri_line_width = 0.10

    cos30 = math.cos(1 / 6)
    sin30 = 0.5

    a = (center[0], center[1] - tri_radius)
    b = (center[0] - tri_radius * cos30, center[1] + tri_radius * sin30)
    c = (center[0] + tri_radius * cos30, center[1] + tri_radius * sin30)

    # fill stroke
    cr.set_line_width(tri_line_width)
    trace_tri(cr, a, b, c)
    cr.stroke()

    # outlines
    cr.set_source_rgb(0, 0, 0)
    cr.set_line_width(outline_line_width)

    tri_line_semiwidth = tri_line_width / 2
    z = 2 * tri_line_semiwidth

    # external outline
    a = (a[0], a[1] - z)
    b = (b[0] - z * cos30, b[1] + z * sin30)
    c = (c[0] + z * cos30, c[1] + z * sin30)
    trace_tri(cr, a, b, c)

    # internal outline
    a = (a[0], a[1] + 2 * z)
    b = (b[0] + 2 * z * cos30, b[1] - 2 * z * sin30)
    c = (c[0] - 2 * z * cos30, c[1] - 2 * z * sin30)
    trace_tri(cr, a, b, c)

    cr.stroke()


def draw_downtri(cr, center, outline_line_width):
    tri_diameter = 0.4
    tri_radius = tri_diameter / 2
    tri_line_width = 0.10

    cos30 = math.cos(1 / 6)
    sin30 = 0.5

    a = (center[0], center[1] + tri_radius)
    b = (center[0] + tri_radius * cos30, center[1] - tri_radius * sin30)
    c = (center[0] - tri_radius * cos30, center[1] - tri_radius * sin30)

    # fill stroke
    cr.set_line_width(tri_line_width)
    trace_tri(cr, a, b, c)
    cr.stroke()

    # outlines
    cr.set_source_rgb(0, 0, 0)
    cr.set_line_width(outline_line_width)

    tri_line_semiwidth = tri_line_width / 2
    z = 2 * tri_line_semiwidth

    # external outline
    a = (a[0], a[1] - z)
    b = (b[0] - z * cos30, b[1] + z * sin30)
    c = (c[0] + z * cos30, c[1] + z * sin30)
    trace_tri(cr, a, b, c)

    # internal outline
    a = (a[0], a[1] + 2 * z)
    b = (b[0] + 2 * z * cos30, b[1] - 2 * z * sin30)
    c = (c[0] - 2 * z * cos30, c[1] - 2 * z * sin30)
    trace_tri(cr, a, b, c)

    cr.stroke()


def trace_x(cr, a1, a2, b1, b2):
    cr.move_to(a1[0], a1[1])
    cr.line_to(a2[0], a2[1])
    cr.move_to(b1[0], b1[1])
    cr.line_to(b2[0], b2[1])


def draw_plus(cr: cairo.Context, center, outline_line_width):
    plus_width = 0.5
    plus_semiwidth = plus_width / 2
    line_width = 0.12
    l = center[0] - plus_semiwidth
    r = center[0] + plus_semiwidth
    t = center[1] - plus_semiwidth
    b = center[1] + plus_semiwidth

    # black
    source = cr.get_source()
    cr.set_source_rgb(0, 0, 0)
    cr.set_line_width(line_width)
    trace_x(cr, (center[0], t), (center[0], b), (l, center[1]), (r, center[1]))
    cr.stroke()

    # fill
    cr.set_source(source)
    cr.set_line_width(line_width - outline_line_width * 2)
    trace_x(cr, (center[0], t), (center[0], b), (l, center[1]), (r, center[1]))

    cr.stroke()


def draw_cross(cr: cairo.Context, center, outline_line_width):
    plus_width = 0.35
    plus_semiwidth = plus_width / 2
    line_width = 0.12
    l = center[0] - plus_semiwidth
    r = center[0] + plus_semiwidth
    t = center[1] - plus_semiwidth
    b = center[1] + plus_semiwidth

    # black
    source = cr.get_source()
    cr.set_source_rgb(0, 0, 0)
    cr.set_line_width(line_width)
    trace_x(cr, (l, t), (r, b), (l, b), (r, t))
    cr.stroke()

    # fill
    cr.set_source(source)
    cr.set_line_width(line_width - outline_line_width * 2)
    trace_x(cr, (l, t), (r, b), (l, b), (r, t))

    cr.stroke()


def draw_dot(cr: cairo.Context, symbol):
    radius = 0.05
    num = b64url_char_to_int(symbol)
    offset = 1 / 16
    offsets = [(0, offset), (0, -offset), (offset, 0), (-offset, 0)]

    num, row = divmod(num, 4)
    num, col = divmod(num, 4)
    sector = offsets[num]

    x = 1 / 8 + row / 4 + sector[0]
    y = 1 / 8 + col / 4 + sector[1]

    cr.set_source_rgb(0, 0, 0)
    cr.arc(x, y, radius, 0, 2 * math.pi)
    cr.fill()


# Initialization
colors = list(itertools.product([0.2, 0.4, 0.6, 0.8], repeat=3))
circles = get_circles()
centers = get_centers()


def draw_hash(cr: cairo.Context, b64str: str, width, height):
    cr.scale(width, height)
    outline_line_width = 0.01
    hl = len(b64str)

    # idx 0 is background color
    if hl < 1:
        return
    set_color(-1, cr, b64str, colors)
    cr.rectangle(0, 0, 1, 1)
    cr.fill()

    # idx 1 is circle position and idx 2 circle color
    idx = 1
    if hl < (idx + 1):
        return
    circle = get_position(circles, b64str, idx)
    draw_circle(cr, circle, idx, b64str, colors, outline_line_width)

    # idx 3 is square position and idx 4 is square color
    idx = 3
    if hl < (idx + 1):
        return
    set_color(idx, cr, b64str, colors)
    center = get_position(centers, b64str, idx)
    draw_square(cr, center, outline_line_width)

    # idx 5 is diamond position and idx 6 is diamond color
    idx = 5
    if hl < (idx + 1):
        return
    set_color(idx, cr, b64str, colors)
    center = get_position(centers, b64str, idx)
    draw_diamond(cr, center, outline_line_width)

    # idx 7  is uptri position and idx 8 is uptri color
    idx = 7
    if hl < (idx + 1):
        return
    set_color(idx, cr, b64str, colors)
    center = get_position(centers, b64str, idx)
    draw_uptri(cr, center, outline_line_width)

    # idx 9  is uptri position and idx 10 is uptri color
    idx = 9
    if hl < (idx + 1):
        return
    set_color(idx, cr, b64str, colors)
    center = get_position(centers, b64str, idx)
    draw_downtri(cr, center, outline_line_width)

    # idx 11  is plus position and idx 12 is plus color
    idx = 11
    if hl < (idx + 1):
        return
    set_color(idx, cr, b64str, colors)
    center = get_position(centers, b64str, idx)
    draw_plus(cr, center, outline_line_width)

    # idx 13  is cross position and idx 14 is cross color
    idx = 13
    if hl < (idx + 1):
        return
    set_color(idx, cr, b64str, colors)
    center = get_position(centers, b64str, idx)
    draw_cross(cr, center, outline_line_width)

    # idx 15 is center symbol, just black
    idx = 15
    if hl < (idx + 1):
        return
    draw_dot(cr, b64str[idx])


def sentence_to_hash(st):
    hash_object = hashlib.sha256(st.encode("utf-8"))
    hash_bytes = hash_object.digest()

    b64_encoded = base64.urlsafe_b64encode(hash_bytes)
    b64hex_str = b64_encoded.decode("utf-8")

    return b64hex_str


def progressive_hash(b64str, display=False):
    for i in range(1, 17):
        disp_and_save(
            draw_hash,
            b64str[:i],
            filename_png=b64str[:i],
            display=display,
        )


def progressive_hashes_unique_start(examples=1, display=False):
    """To preserve filesystem order"""
    if examples > 38:
        raise ValueError(
            f"Impossible to have {examples} case-insensitive unique starts"
        )
    starts = set()
    example = 0
    while example < examples:
        b64str = "".join(random.choices(B64URL, k=16))
        if not "-" in b64str and not "_" in b64str and b64str[0].lower() not in starts:
            print(b64str)
            starts.add(b64str[0].lower())
            progressive_hash(b64str, display=display)
            example += 1


def random_hash_drawings(num=1,to_png=True,to_svg=False,display=False):
    for i in range(num):
        b64str = "".join(random.choices(B64URL, k=16))
        print(b64str)
        if to_png:
            if to_svg:
                disp_and_save(draw_hash, b64str, filename_png=b64str, filename_svg=b64str, display=display)
            else:
                disp_and_save(draw_hash, b64str, filename_png=b64str, display=display)
        else:
            if to_svg:
                disp_and_save(draw_hash, b64str, filename_svg=b64str, display=display)
            else:
                disp_and_save(draw_hash, b64str, display=display)


#progressive_hashes_unique_start(1)
#random_hash_drawings(num=2000)

# TODO put it to use
disp_and_save(draw_hash, "he", filename_png="he", display=True)
disp_and_save(draw_hash, "hello", filename_png="hello", display=True)

random.seed(0)
progressive_hashes_unique_start(1)
random_hash_drawings(num=10)