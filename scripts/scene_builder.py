"""
scene_builder.py — tabletop PyBullet scene with YCB + primitive objects.

Rendering pipeline:
  1. Three-point lighting (key + fill + rim) blended in float32
  2. SSAA x3 supersampling
  3. Post-processing: UnsharpMask, contrast, saturation, gamma, vignette
"""

import os
import pybullet as p
import pybullet_data
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from PIL import Image as PILImage, ImageEnhance, ImageFilter

YCB_CATALOG = [
    ("YcbBanana",          "banana",           0.045, 0.030),
    ("YcbTomatoSoupCan",   "tomato_soup_can",  0.051, 0.033),
    ("YcbMustardBottle",   "mustard_bottle",   0.095, 0.030),
    ("YcbCrackerBox",      "cracker_box",      0.090, 0.040),
    ("YcbGelatinBox",      "gelatin_box",      0.035, 0.045),
    ("YcbPottedMeatCan",   "meat_can",         0.042, 0.040),
    ("YcbMasterChefCan",   "master_chef_can",  0.070, 0.038),
    ("YcbFoamBrick",       "foam_brick",       0.030, 0.038),
    ("YcbStrawberry",      "strawberry",       0.022, 0.018),
    ("YcbPear",            "pear",             0.040, 0.025),
    ("YcbTennisBall",      "tennis_ball",      0.033, 0.033),
    ("YcbChipsCan",        "chips_can",        0.100, 0.033),
    ("YcbMarker",          "marker",           0.060, 0.010),
    ("YcbMediumClamp",     "clamp",            0.030, 0.040),
]

_SPECULAR = {
    "cube":     [0.6, 0.6, 0.6],
    "cylinder": [0.8, 0.8, 0.8],
    "sphere":   [1.0, 1.0, 1.0],
}


@dataclass
class ObjectMeta:
    obj_id:      str
    bullet_id:   int
    category:    str
    color_name:  str
    color_rgba:  List[float]
    position:    Tuple[float, float, float]
    half_height: float
    source:      str


# ------------------------------------------------------------------
# Post-processing
# ------------------------------------------------------------------

def postprocess(img: np.ndarray) -> np.ndarray:
    pil = PILImage.fromarray(img)

    pil = pil.filter(ImageFilter.UnsharpMask(radius=1.2, percent=140, threshold=2))
    pil = ImageEnhance.Contrast(pil).enhance(1.15)
    pil = ImageEnhance.Color(pil).enhance(1.20)
    pil = ImageEnhance.Brightness(pil).enhance(1.30)   # lift overall exposure

    arr = np.array(pil, dtype=np.float32) / 255.0
    arr = np.power(arr, 0.88)                          # gamma — lift midtones

    # Soft vignette — gentle corner darkening only
    h, w = arr.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(((X - w/2) / (w/2))**2 + ((Y - h/2) / (h/2))**2)
    vignette = 1.0 - np.clip(dist * 0.28, 0, 0.22)
    arr *= vignette[:, :, np.newaxis]

    return np.clip(arr * 255, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------
# Scene builder
# ------------------------------------------------------------------

class SceneBuilder:
    TABLE_HEIGHT    = 0.70
    TABLE_THICKNESS = 0.04
    TABLE_HALF      = 0.35
    LEG_RADIUS      = 0.025
    SSAA            = 3      # render 1920x1440 → downsample to 640x480

    def __init__(self, width: int = 640, height: int = 480, use_gui: bool = False):
        self.width  = width
        self.height = height
        self._use_gui = use_gui
        self.physics_client = None
        self._obj_counter   = 0
        self._ycb_path      = self._find_ycb_path()

    @staticmethod
    def _find_ycb_path():
        try:
            from pybullet_object_models import ycb_objects
            return ycb_objects.getDataPath()
        except ImportError:
            return None

    def ycb_available(self):
        return self._ycb_path is not None

    def available_ycb_objects(self):
        if not self._ycb_path:
            return []
        result = []
        for folder, name, hh, hr in YCB_CATALOG:
            urdf = os.path.join(self._ycb_path, folder, "model.urdf")
            if os.path.exists(urdf):
                result.append({"folder": folder, "name": name, "urdf": urdf,
                                "half_height": hh, "half_radius": hr})
        return result

    def connect(self):
        mode = p.GUI if self._use_gui else p.DIRECT
        self.physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

    def disconnect(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self._obj_counter = 0

    def build(self, object_specs, camera_yaw=45.0,
              camera_pitch=-35.0, camera_distance=1.0):
        self.reset()
        self._load_environment()
        objects = self._place_objects(object_specs)
        self._settle_physics(steps=200)
        self._refresh_positions(objects)
        rgb = self._render(camera_yaw, camera_pitch, camera_distance)
        rgb = postprocess(rgb)
        return rgb, objects

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def _load_environment(self):
        # Dark floor
        fc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[4, 4, 0.01])
        fv = p.createVisualShape(p.GEOM_BOX, halfExtents=[4, 4, 0.01],
                                  rgbaColor=[0.12, 0.12, 0.14, 1.0])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=fc,
                          baseVisualShapeIndex=fv, basePosition=[0, 0, -0.01])

        # Table surface
        th = self.TABLE_THICKNESS / 2
        tc = p.createCollisionShape(p.GEOM_BOX,
                                     halfExtents=[self.TABLE_HALF, self.TABLE_HALF, th])
        tv = p.createVisualShape(p.GEOM_BOX,
                                  halfExtents=[self.TABLE_HALF, self.TABLE_HALF, th],
                                  rgbaColor=[0.72, 0.52, 0.32, 1.0],
                                  specularColor=[0.3, 0.25, 0.15])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=tc,
                          baseVisualShapeIndex=tv,
                          basePosition=[0, 0, self.TABLE_HEIGHT])

        # Table legs
        lh = (self.TABLE_HEIGHT - self.TABLE_THICKNESS / 2) / 2
        for lx, ly in [(self.TABLE_HALF-0.05,  self.TABLE_HALF-0.05),
                        (-self.TABLE_HALF+0.05,  self.TABLE_HALF-0.05),
                        (self.TABLE_HALF-0.05,  -self.TABLE_HALF+0.05),
                        (-self.TABLE_HALF+0.05, -self.TABLE_HALF+0.05)]:
            lc = p.createCollisionShape(p.GEOM_CYLINDER,
                                         radius=self.LEG_RADIUS, height=lh * 2)
            lv = p.createVisualShape(p.GEOM_CYLINDER,
                                      radius=self.LEG_RADIUS, length=lh * 2,
                                      rgbaColor=[0.52, 0.36, 0.20, 1.0])
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=lc,
                              baseVisualShapeIndex=lv, basePosition=[lx, ly, lh])

        # Back wall — subtle depth cue
        wc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.01, 0.5])
        wv = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.01, 0.5],
                                  rgbaColor=[0.22, 0.22, 0.25, 1.0])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wc,
                          baseVisualShapeIndex=wv,
                          basePosition=[0, self.TABLE_HALF + 0.01,
                                        self.TABLE_HEIGHT + 0.5])

    # ------------------------------------------------------------------
    # Objects
    # ------------------------------------------------------------------

    def _place_objects(self, specs):
        objects = []
        surface_z = self.TABLE_HEIGHT + self.TABLE_THICKNESS / 2
        for spec in specs:
            xy  = spec["position_xy"]
            src = spec.get("source", "primitive")
            obj = (self._place_ycb(spec, xy, surface_z) if src == "ycb"
                   else self._place_primitive(spec, xy, surface_z))
            if obj is not None:
                objects.append(obj)
        return objects

    def _place_ycb(self, spec, xy, surface_z):
        hh = spec["half_height"]
        z  = surface_z + hh + 0.002
        try:
            bid = p.loadURDF(spec["urdf"],
                             basePosition=[xy[0], xy[1], z],
                             baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                             globalScaling=1.0)
        except Exception as e:
            print(f"[WARN] YCB load failed ({spec['name']}): {e}")
            return None

        # ER_TINY_RENDERER ignores .mtl textures — apply per-link fallback colors
        YCB_FALLBACK_COLORS = {
            "banana":           [0.95, 0.85, 0.10, 1.0],
            "tomato_soup_can":  [0.80, 0.15, 0.10, 1.0],
            "mustard_bottle":   [0.90, 0.75, 0.05, 1.0],
            "cracker_box":      [0.85, 0.55, 0.20, 1.0],
            "gelatin_box":      [0.90, 0.20, 0.50, 1.0],
            "meat_can":         [0.70, 0.65, 0.55, 1.0],
            "master_chef_can":  [0.15, 0.25, 0.65, 1.0],
            "foam_brick":       [0.95, 0.40, 0.10, 1.0],
            "strawberry":       [0.90, 0.10, 0.15, 1.0],
            "pear":             [0.70, 0.85, 0.20, 1.0],
            "tennis_ball":      [0.75, 0.90, 0.15, 1.0],
            "chips_can":        [0.85, 0.65, 0.10, 1.0],
            "marker":           [0.20, 0.20, 0.80, 1.0],
            "clamp":            [0.60, 0.60, 0.60, 1.0],
        }
        fallback = YCB_FALLBACK_COLORS.get(spec["name"], [0.70, 0.70, 0.70, 1.0])
        # getVisualShapeData returns all links that have visuals — most reliable way
        for vd in p.getVisualShapeData(bid):
            link_idx = vd[1]   # -1 = base link, 0..N = child links
            p.changeVisualShape(bid, link_idx, rgbaColor=fallback)
        obj_id = f"obj_{self._obj_counter:03d}"
        self._obj_counter += 1
        return ObjectMeta(obj_id=obj_id, bullet_id=bid,
                          category=spec["name"], color_name="n/a",
                          color_rgba=[0, 0, 0, 0],
                          position=(xy[0], xy[1], z),
                          half_height=hh, source="ycb")

    def _place_primitive(self, spec, xy, surface_z):
        cat  = spec["category"]
        size = spec["size"]
        rgba = spec["color_rgba"]
        half = [s / 2 for s in size]
        z    = surface_z + half[2] + 0.001
        sc   = _SPECULAR.get(cat, [0.5, 0.5, 0.5])
        if cat == "cube":
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                       rgbaColor=rgba, specularColor=sc)
        elif cat == "cylinder":
            col = p.createCollisionShape(p.GEOM_CYLINDER,
                                          radius=half[0], height=size[2])
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=half[0],
                                       length=size[2], rgbaColor=rgba, specularColor=sc)
        elif cat == "sphere":
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=half[0])
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=half[0],
                                       rgbaColor=rgba, specularColor=sc)
        else:
            return None
        bid = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=col,
                                baseVisualShapeIndex=vis,
                                basePosition=[xy[0], xy[1], z])
        obj_id = f"obj_{self._obj_counter:03d}"
        self._obj_counter += 1
        return ObjectMeta(obj_id=obj_id, bullet_id=bid,
                          category=f"{spec['color_name']}_{cat}",
                          color_name=spec["color_name"], color_rgba=rgba,
                          position=(xy[0], xy[1], z),
                          half_height=half[2], source="primitive")

    def _settle_physics(self, steps=200):
        for _ in range(steps):
            p.stepSimulation()

    def _refresh_positions(self, objects):
        for obj in objects:
            pos, _ = p.getBasePositionAndOrientation(obj.bullet_id)
            obj.position = tuple(pos)

    # ------------------------------------------------------------------
    # Three-point lighting render
    # ------------------------------------------------------------------

    def _render_pass(self, W, H, view, proj,
                     light_dir, light_color, ambient, diffuse, specular):
        _, _, rgba, _, _ = p.getCameraImage(
            width=W, height=H,
            viewMatrix=view, projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
            lightDirection=light_dir,
            lightColor=light_color,
            lightDistance=3.0,
            shadow=1,
            lightAmbientCoeff=ambient,
            lightDiffuseCoeff=diffuse,
            lightSpecularCoeff=specular,
        )
        return np.array(rgba, dtype=np.float32)[:, :, :3]

    def _render(self, yaw, pitch, distance):
        W = self.width  * self.SSAA
        H = self.height * self.SSAA

        target = [0, 0, self.TABLE_HEIGHT + self.TABLE_THICKNESS / 2]
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target, distance=distance,
            yaw=yaw, pitch=pitch, roll=0, upAxisIndex=2)
        proj = p.computeProjectionMatrixFOV(
            fov=52, aspect=W / H, nearVal=0.01, farVal=10.0)

        # Key light — warm, upper-front-right
        key  = self._render_pass(W, H, view, proj,
                                  [0.6,  0.4, 1.0], [1.00, 0.95, 0.88],
                                  0.45, 0.85, 0.40)
        # Fill light — cool, left, no hard shadow
        fill = self._render_pass(W, H, view, proj,
                                  [-0.8, 0.2, 0.6], [0.85, 0.90, 1.00],
                                  0.65, 0.40, 0.05)
        # Rim light — back edge separation
        rim  = self._render_pass(W, H, view, proj,
                                  [0.0, -1.0, 0.4], [0.95, 0.95, 1.00],
                                  0.35, 0.25, 0.10)

        # Blend: 60% key + 28% fill + 12% rim
        blended = key * 0.60 + fill * 0.28 + rim * 0.12
        img = np.clip(blended, 0, 255).astype(np.uint8)

        # SSAA downsample
        pil = PILImage.fromarray(img)
        pil = pil.resize((self.width, self.height), PILImage.LANCZOS)
        return np.array(pil)
