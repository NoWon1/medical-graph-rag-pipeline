from PIL import Image
import numpy as np

def original_color(img: Image.Image) -> dict:
    rgb   = img.convert("RGB")
    w, h  = rgb.size
    total = w * h
    if total == 0:
        return {k: 0 for k in ["bw_ratio","green_ratio","teal_ratio",
                                "orange_ratio","sepia_ratio",
                                "dominant_hue_frac","edge_ratio"]}
    pixels = list(rgb.getdata())
    BW_THRESH   = 30
    bw_count = green_count = teal_count = orange_count = sepia_count = 0
    hue_buckets = [0] * 36

    for r, g, b in pixels:
        lo, hi = min(r, g, b), max(r, g, b)

        if (hi - lo) < BW_THRESH and (hi < 50 or lo > 205):
            bw_count += 1

        if r < 120 and 160 <= g <= 230 and b < 120:
            green_count += 1

        if r < 100 and g > 150 and b > 150 and abs(g - b) < 40:
            teal_count += 1
        elif r > 180 and 80 <= g <= 160 and b < 80:
            orange_count += 1
        elif 100 <= r <= 210 and 60 <= g <= 150 and 20 <= b <= 110 and r > g > b and (r - b) > 40:
            sepia_count += 1

        delta = hi - lo
        if delta > 40 and hi > 0:
            if hi == r:   hue = (60 * ((g - b) / delta)) % 360
            elif hi == g: hue = 60 * ((b - r) / delta) + 120
            else:         hue = 60 * ((r - g) / delta) + 240
            hue_buckets[int(hue / 10) % 36] += 1

    sat_total = sum(hue_buckets)
    if sat_total > total * 0.10:
        tb  = max(range(36), key=lambda i: hue_buckets[i])
        tc  = sum(hue_buckets[(tb + d) % 36] for d in [-1, 0, 1])
        dhf = tc / sat_total
    else:
        dhf = 0.0

    grey  = img.convert("L").resize((64, 64), Image.LANCZOS)
    gpix  = list(grey.getdata())
    gw = gh = 64; ec = 0; ET = 30
    for row in range(gh - 1):
        for col in range(gw - 1):
            idx = row * gw + col
            if (abs(int(gpix[idx]) - int(gpix[idx + 1])) > ET or
                    abs(int(gpix[idx]) - int(gpix[idx + gw])) > ET):
                ec += 1
    edge_ratio = ec / (gw * gh)

    return {
        "bw_ratio":          bw_count     / total,
        "green_ratio":       green_count  / total,
        "teal_ratio":        teal_count   / total,
        "orange_ratio":      orange_count / total,
        "sepia_ratio":       sepia_count  / total,
        "dominant_hue_frac": dhf,
        "edge_ratio":        edge_ratio,
    }


def vectorized_color(img: Image.Image) -> dict:
    rgb   = img.convert("RGB")
    w, h  = rgb.size
    total = w * h
    if total == 0:
        return {k: 0 for k in ["bw_ratio","green_ratio","teal_ratio",
                                "orange_ratio","sepia_ratio",
                                "dominant_hue_frac","edge_ratio"]}

    # ⚡ Bolt: Vectorized pixel math to avoid slow generator iteration and underflow bugs
    arr = np.asarray(rgb, dtype=np.int16)
    r = arr[:, :, 0].flatten()
    g = arr[:, :, 1].flatten()
    b = arr[:, :, 2].flatten()

    lo = np.minimum(np.minimum(r, g), b)
    hi = np.maximum(np.maximum(r, g), b)
    delta = hi - lo

    bw_count = int(np.count_nonzero((delta < 30) & ((hi < 50) | (lo > 205))))
    green_count = int(np.count_nonzero((r < 120) & (g >= 160) & (g <= 230) & (b < 120)))
    teal_mask = (r < 100) & (g > 150) & (b > 150) & (np.abs(g - b) < 40)
    teal_count = int(np.count_nonzero(teal_mask))
    orange_mask = (r > 180) & (g >= 80) & (g <= 160) & (b < 80)
    orange_count = int(np.count_nonzero(orange_mask))
    sepia_mask = (~orange_mask) & (r >= 100) & (r <= 210) & (g >= 60) & (g <= 150) & (b >= 20) & (b <= 110) & (r > g) & (g > b) & ((r - b) > 40)
    sepia_count = int(np.count_nonzero(sepia_mask))

    sat_mask = (delta > 40) & (hi > 0)
    hue = np.zeros_like(r, dtype=np.float64)
    valid_delta = np.where(delta == 0, 1, delta)

    m_r = sat_mask & (hi == r)
    hue[m_r] = (60 * ((g[m_r] - b[m_r]) / valid_delta[m_r])) % 360

    m_g = sat_mask & ~m_r & (hi == g)
    hue[m_g] = 60 * ((b[m_g] - r[m_g]) / valid_delta[m_g]) + 120

    m_b = sat_mask & ~m_r & ~m_g
    hue[m_b] = 60 * ((r[m_b] - g[m_b]) / valid_delta[m_b]) + 240

    if np.any(sat_mask):
        valid_hues = hue[sat_mask]
        bins = (valid_hues / 10).astype(int) % 36
        hue_buckets = np.bincount(bins, minlength=36).tolist()
    else:
        hue_buckets = [0] * 36

    sat_total = sum(hue_buckets)
    if sat_total > total * 0.10:
        tb  = max(range(36), key=lambda i: hue_buckets[i])
        tc  = sum(hue_buckets[(tb + d) % 36] for d in [-1, 0, 1])
        dhf = tc / sat_total
    else:
        dhf = 0.0

    grey = img.convert("L").resize((64, 64), Image.LANCZOS)
    garr = np.asarray(grey, dtype=np.int16)
    diff_x = np.abs(garr[:-1, :-1] - garr[:-1, 1:])
    diff_y = np.abs(garr[:-1, :-1] - garr[1:, :-1])
    ec = int(np.count_nonzero((diff_x > 30) | (diff_y > 30)))
    edge_ratio = ec / (64 * 64)

    return {
        "bw_ratio":          bw_count     / total,
        "green_ratio":       green_count  / total,
        "teal_ratio":        teal_count   / total,
        "orange_ratio":      orange_count / total,
        "sepia_ratio":       sepia_count  / total,
        "dominant_hue_frac": dhf,
        "edge_ratio":        edge_ratio,
    }


def run_tests():
    import random
    # create dummy image
    img = Image.new('RGB', (100, 100))
    pixels = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixels[i, j] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

    res1 = original_color(img)
    res2 = vectorized_color(img)

    print("Match?", res1 == res2)
    for k in res1:
        if res1[k] != res2[k]:
            print(k, res1[k], res2[k])

if __name__ == "__main__":
    run_tests()
