# ==========================================
# 🔹 BACKEND PYTHON - SEGMENTACIÓN MÉDICA
# ==========================================

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import numpy as np
from skimage import io, morphology, feature, color, exposure, util, segmentation
from skimage.restoration import denoise_bilateral
from skimage.filters import threshold_local
from skimage.measure import regionprops
from scipy import ndimage as ndi
from PIL import Image
from io import BytesIO

# ==========================================
# 🔹 FUNCIONES DE SEGMENTACIÓN
# ==========================================

def local_variance(image, size=9):
    mean = ndi.uniform_filter(image, size=size)
    mean_sq = ndi.uniform_filter(image ** 2, size=size)
    return mean_sq - mean ** 2

def filter_by_area(labels, min_area, max_area):
    props = regionprops(labels)
    filtered = np.zeros_like(labels)
    next_label = 1
    for region in props:
        if min_area <= region.area <= max_area:
            filtered[labels == region.label] = next_label
            next_label += 1
    return filtered

def watershed_segmentation(image, params):
    if len(image.shape) == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image

    eq = exposure.equalize_adapthist(gray, clip_limit=0.03)
    denoised = denoise_bilateral(eq, sigma_color=0.05, sigma_spatial=10, channel_axis=None)
    texture_map = local_variance(denoised, size=9)
    texture_map /= np.max(texture_map)
    combined = np.clip(denoised + params["weight_texture"] * texture_map, 0, 1)
    edges = feature.canny(combined, sigma=params["sigma_canny"])

    if params["use_threshold"]:
        local_thresh = threshold_local(combined, block_size=int(params["block_size"]), offset=params["offset"])
        binary = combined > local_thresh
        binary = np.logical_or(binary, edges)
    else:
        binary = np.ones(combined.shape, dtype=bool)
        binary = np.logical_or(binary, edges)

    binary = morphology.remove_small_objects(binary, min_size=int(params["min_area"]))
    binary = morphology.remove_small_holes(binary, area_threshold=500)
    binary = morphology.closing(binary, morphology.disk(3))

    dist = ndi.distance_transform_edt(binary)
    local_maxi = feature.peak_local_max(dist, min_distance=int(params["min_distance"]), labels=binary)
    mask = np.zeros(dist.shape, dtype=bool)
    if local_maxi.size > 0:
        mask[tuple(local_maxi.T)] = True

    markers, _ = ndi.label(mask)
    labels = segmentation.watershed(-dist, markers, mask=binary)
    labels = filter_by_area(labels, int(params["min_area"]), int(params["max_area"]))

    return labels, eq, texture_map

# ==========================================
# 🔹 SERVIDOR FLASK
# ==========================================

app = Flask(_name_)
CORS(app)

@app.route('/')
def home():
    return "✅ Servidor activo: listo para segmentar imágenes médicas."

@app.route('/segment', methods=['POST'])
def segment_image():
    try:
        file = request.files['image']
        img = io.imread(file)

        params = {
            "min_distance": float(request.form.get('min_distance', 20)),
            "min_area": float(request.form.get('min_area', 100)),
            "max_area": float(request.form.get('max_area', 10000)),
            "block_size": float(request.form.get('block_size', 51)),
            "offset": float(request.form.get('offset', 0.02)),
            "sigma_canny": float(request.form.get('sigma_canny', 1.5)),
            "weight_texture": float(request.form.get('weight_texture', 0.3)),
            "use_threshold": request.form.get('use_threshold', 'true').lower() == 'true'
        }

        labels, eq, texture_map = watershed_segmentation(img, params)
        overlay = color.label2rgb(labels, image=img, bg_label=0, kind='overlay', alpha=0.4)

        result = Image.fromarray((overlay * 255).astype(np.uint8))
        buf = BytesIO()
        result.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if _name_ == '__main__':

    app.run(host='0.0.0.0', port=5000)
