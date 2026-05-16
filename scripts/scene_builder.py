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
import math
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
    "cube":     [0.12, 0.12, 0.12],
    "cylinder": [0.18, 0.18, 0.18],
    "sphere":   [0.25, 0.25, 0.25],
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

    # slight cinematic blur
    pil = pil.filter(
        ImageFilter.GaussianBlur(radius=0.35)
    )

    # soft sharpen
    pil = pil.filter(
        ImageFilter.UnsharpMask(
            radius=0.8,
            percent=70,
            threshold=3,
        )
    )

    # mild grading
    pil = ImageEnhance.Contrast(pil).enhance(1.05)
    pil = ImageEnhance.Color(pil).enhance(1.06)
    pil = ImageEnhance.Brightness(pil).enhance(1.04)

    arr = np.array(pil, dtype=np.float32) / 255.0

    # -------------------------------------------------
    # FILMIC ACES TONEMAP
    # -------------------------------------------------

    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14

    arr = np.clip(
        (arr * (a * arr + b)) /
        (arr * (c * arr + d) + e),
        0,
        1,
    )

    # slight gamma
    arr = np.power(arr, 0.98)

    h, w = arr.shape[:2]

    # -------------------------------------------------
    # ATMOSPHERIC HAZE
    # -------------------------------------------------

    fog = np.linspace(1.0, 0.94, h)[:, None, None]
    arr *= fog

    # -------------------------------------------------
    # VIGNETTE
    # -------------------------------------------------

    Y, X = np.ogrid[:h, :w]

    dist = np.sqrt(
        ((X - w/2) / (w/2))**2 +
        ((Y - h/2) / (h/2))**2
    )

    vignette = 1.0 - np.clip(dist * 0.10, 0, 0.06)

    arr *= vignette[:, :, None]

    # -------------------------------------------------
    # SENSOR GRAIN
    # -------------------------------------------------

    grain = np.random.normal(0, 0.004, arr.shape)
    arr = np.clip(arr + grain, 0, 1)

    # -------------------------------------------------
    # WOOD / TEXTURE VARIATION
    # -------------------------------------------------

    texture_noise = np.random.normal(1.0, 0.01, (h, w))
    return (arr * 255).astype(np.uint8)


# ------------------------------------------------------------------
# Scene builder
# ------------------------------------------------------------------

class SceneBuilder:
    TABLE_HEIGHT    = 0.625  # top surface: origin z=0.6 + half thickness 0.025
    TABLE_THICKNESS = 0.05
    TABLE_HALF      = 0.75   # half of 1.5 scale in x
    LEG_RADIUS      = 0.025
    SSAA            = 2      # render 1920x1440 → downsample to 640x480

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

    def build(
        self,
        object_specs,
        camera_yaw=45.0,
        camera_pitch=-35.0,
        camera_distance=1.0,
    ):
        MAX_RETRIES = 5

        last_error = None

        for attempt in range(MAX_RETRIES):

            try:
                self.reset()

                self._load_environment()

                objects = self._place_objects(object_specs)

                # if all objects failed
                if len(objects) == 0:
                    raise RuntimeError("No objects placed")

                self._settle_physics(steps=200)

                self._refresh_positions(objects)

                rgb = self._render(
                    camera_yaw,
                    camera_pitch,
                    camera_distance,
                )

                # detect broken render
                mean_intensity = rgb.mean()

                # black frame
                if mean_intensity < 3:
                    raise RuntimeError(
                        f"Black frame detected ({mean_intensity:.2f})"
                    )

                # NaNs
                if not np.isfinite(rgb).all():
                    raise RuntimeError("NaN render detected")

                rgb = postprocess(rgb)

                return rgb, objects

            except Exception as e:

                last_error = e

                print(
                    f"[WARN] Scene build failed "
                    f"(attempt {attempt+1}/{MAX_RETRIES}): {e}"
                )

        raise RuntimeError(
            f"Scene generation failed after retries: {last_error}"
        )

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def _load_environment(self):
        # Dark floor
        fc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[4, 4, 0.01])
        fv = p.createVisualShape(p.GEOM_BOX, halfExtents=[4, 4, 0.01],
                                  rgbaColor=[0.22, 0.22, 0.24, 1.0])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=fc,
                          baseVisualShapeIndex=fv, basePosition=[0, 0, -0.01])

        # Custom table URDF (table.urdf + table.obj + table.png)
        table_urdf = os.path.join(
            os.path.dirname(__file__), "assets", "table", "table.urdf"
        )
        if os.path.exists(table_urdf):
            # Add assets dir to search path so PyBullet finds .obj and .png
            p.setAdditionalSearchPath(
                os.path.join(os.path.dirname(__file__), "assets", "table")
            )
            p.loadURDF(table_urdf, basePosition=[0, 0, 0], useFixedBase=True)
            self._using_custom_table = True
        else:
            # Fallback: procedural table
            print("[WARN] table.urdf not found in scripts/assets/table/ — using procedural table")
            self._using_custom_table = False
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

        # Back wall
        wc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1.5, 0.02, 1.2])
        wv = p.createVisualShape(p.GEOM_BOX, halfExtents=[1.5, 0.02, 1.2],
                                  rgbaColor=[0.42, 0.42, 0.45, 1.0])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wc,
                          baseVisualShapeIndex=wv,
                          basePosition=[0, self.TABLE_HALF + 0.02,
                                        self.TABLE_HEIGHT - 0.2])

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
            link_idx = vd[1]

            p.changeVisualShape(
                bid,
                link_idx,

                rgbaColor=fallback,

                specularColor=[0.08, 0.08, 0.08],

                textureUniqueId=-1,
            )


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

    def _render_pass(
        self,
        W,
        H,
        view,
        proj,
        light_dir,
        light_color,
        ambient,
        diffuse,
        specular,
        shadow=True,
    ):

        _, _, rgba, _, _ = p.getCameraImage(
            width=W,
            height=H,
            viewMatrix=view,
            projectionMatrix=proj,

            renderer=p.ER_TINY_RENDERER,

            lightDirection=light_dir,
            lightColor=light_color,
            lightDistance=3.0,

            shadow=1 if shadow else 0,

            lightAmbientCoeff=ambient,
            lightDiffuseCoeff=diffuse,
            lightSpecularCoeff=specular,
        )

        arr = np.array(rgba, dtype=np.float32)[:, :, :3]

        # broken renderer protection
        if arr.mean() < 1:
            raise RuntimeError("Renderer returned black frame")

        return arr

    def _render(self, yaw, pitch, distance):

        distance = max(0.7, min(distance, 1.8))
        pitch = max(-80, min(pitch, -10))

        W = self.width * self.SSAA
        H = self.height * self.SSAA

        target = [0, 0, self.TABLE_HEIGHT + self.TABLE_THICKNESS / 2]

        # -------------------------------------------------
        # RANDOMIZED CAMERA
        # -------------------------------------------------

        yaw += np.random.normal(0, 1.0)
        pitch += np.random.normal(0, 0.8)
        distance += np.random.normal(0, 0.025)

        proj = p.computeProjectionMatrixFOV(
            fov=44,
            aspect=W / H,
            nearVal=0.01,
            farVal=10.0,
        )

        # -------------------------------------------------
        # TEMPORAL AA
        # -------------------------------------------------

        samples = []

        TAA_SAMPLES = 4

        for _ in range(TAA_SAMPLES):

            jitter_yaw = yaw + np.random.normal(0, 0.08)
            jitter_pitch = pitch + np.random.normal(0, 0.08)

            view = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=target,
                distance=distance,
                yaw=jitter_yaw,
                pitch=jitter_pitch,
                roll=0,
                upAxisIndex=2,
            )

            # -------------------------------------------------
            # RANDOMIZED LIGHTING
            # -------------------------------------------------

            key_strength = np.random.uniform(0.96, 1.04)
            fill_strength = np.random.uniform(0.96, 1.04)

            key = self._render_pass(
                W,
                H,
                view,
                proj,
                light_dir=[
                    0.55 + np.random.normal(0, 0.02),
                    0.35 + np.random.normal(0, 0.02),
                    0.72,
                ],
                light_color=[
                    1.00 * key_strength,
                    0.97 * key_strength,
                    0.92 * key_strength,
                ],
                ambient=0.28,
                diffuse=0.72,
                specular=0.16,
                shadow=True,
            )

            fill = self._render_pass(
                W,
                H,
                view,
                proj,
                light_dir=[
                    -0.65,
                    0.25,
                    0.45,
                ],
                light_color=[
                    0.90 * fill_strength,
                    0.93 * fill_strength,
                    1.00 * fill_strength,
                ],
                ambient=0.22,
                diffuse=0.28,
                specular=0.02,
                shadow=False,
            )

            rim = self._render_pass(
                W,
                H,
                view,
                proj,
                light_dir=[-0.2, -1.0, 0.45],
                light_color=[0.96, 0.96, 1.00],
                ambient=0.05,
                diffuse=0.18,
                specular=0.08,
                shadow=False,
            )

            blended = (
                key * 0.72 +
                fill * 0.20 +
                rim * 0.08
            )

            samples.append(blended.astype(np.float32))

        img = np.mean(samples, axis=0)

        img = np.clip(img, 0, 255).astype(np.uint8)

        # -------------------------------------------------
        # SSAA DOWNSAMPLE
        # -------------------------------------------------

        pil = PILImage.fromarray(img)

        pil = pil.resize(
            (self.width, self.height),
            PILImage.LANCZOS,
        )

        return np.array(pil)
