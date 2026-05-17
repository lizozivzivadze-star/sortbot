"""
game/conveyor.py
----------------
Pygame conveyor belt game.

Objects on the belt are real CIFAR-10 image thumbnails processed
through the vision pipeline in a background thread. The game shows:

  - The thumbnail rolling down the belt
  - A live "pipeline log" panel showing what the CV stages are doing
  - The predicted class + bin above the thumbnail
  - A sort actuator arm swinging to the correct bin
  - Score, accuracy, streak, and per-class breakdown

Controls (keyboard):
  SPACE  — pause / unpause
  1/2/3  — set belt speed
  N      — toggle noise overlay (visual effect only, noise set at launch)
  Q/ESC  — quit
"""

import sys
import math
import threading
import queue
import time
import collections

import pygame
import numpy as np

from core.dataset import get_bin

# ── Colours (flat palette) ───────────────────────────────────────────────────
BG          = (18,  18,  20)
BELT_DARK   = (35,  35,  40)
BELT_STRIPE = (45,  45,  50)
WHITE       = (240, 240, 235)
MUTED       = (130, 128, 120)
GREEN       = (29,  158, 117)
RED         = (226, 75,  74)
AMBER       = (239, 159, 39)
BLUE        = (55,  138, 221)
PURPLE      = (127, 119, 221)

BIN_COLORS = {
    "recycle": BLUE,
    "organic":  GREEN,
    "metal":   PURPLE,
    "reject":  RED,
}

BIN_LABELS = {
    "recycle": "Recycle",
    "organic":  "Organic",
    "metal":   "Metal",
    "reject":  "Reject",
}

FONT_MONO = None   # set after pygame.init()
FONT_SANS = None
FONT_SM   = None
FONT_LG   = None


class ConveyorItem:
    """One image travelling along the belt."""
    __slots__ = [
        "surface", "true_label", "true_bin",
        "x", "y", "classified", "result",
        "age", "done", "flash_timer",
    ]

    def __init__(self, surface, true_label, true_bin, start_x):
        self.surface     = surface
        self.true_label  = true_label
        self.true_bin    = true_bin
        self.x           = float(start_x)
        self.y           = 0.0        # will be set by game
        self.classified  = False
        self.result      = None
        self.age         = 0.0
        self.done        = False
        self.flash_timer = 0.0


class ConveyorGame:
    BELT_Y      = 280   # centre of belt
    BELT_H      = 90
    ITEM_SIZE   = 56    # thumbnail size on belt
    SCAN_X      = 420   # where the vision system scans
    BIN_Y       = 460   # centre of bins row
    BIN_W       = 110
    BIN_H       = 80
    W, H        = 760, 620

    def __init__(self, images, labels, class_names, pipeline, belt_speed=2):
        self.images      = images
        self.labels      = labels
        self.class_names = class_names
        self.pipeline    = pipeline
        self.belt_speed  = belt_speed

        self.items       = []
        self.img_index   = 0
        self.spawn_timer = 0.0
        self.spawn_gap   = 2.2          # seconds between spawns

        self.score       = 0
        self.streak      = 0
        self.best_streak = 0
        self.total       = 0
        self.correct     = 0
        self.per_class   = collections.defaultdict(lambda: [0, 0])  # [correct, total]

        self.paused      = False
        self.belt_offset = 0.0         # for belt stripe animation
        self.pipeline_log= ["Initialising…"]
        self.feedback    = []          # [(text, x, y, colour, alpha, dy)]
        self.arm_angle   = 0.0        # degrees, 0 = centre
        self.arm_target  = 0.0

        # Background classification thread
        self._clf_queue  = queue.Queue()
        self._res_queue  = queue.Queue()
        self._clf_thread = threading.Thread(target=self._classify_worker, daemon=True)
        self._clf_thread.start()

        # Results tracking
        self.results     = {}

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self):
        pygame.init()
        pygame.display.set_caption("SortBot — Vision Conveyor")

        global FONT_MONO, FONT_SANS, FONT_SM, FONT_LG
        FONT_MONO = pygame.font.SysFont("monospace",    13)
        FONT_SANS = pygame.font.SysFont("sans-serif",   14)
        FONT_SM   = pygame.font.SysFont("sans-serif",   12)
        FONT_LG   = pygame.font.SysFont("sans-serif",   22, bold=True)

        screen = pygame.display.set_mode((self.W, self.H))
        clock  = pygame.time.Clock()

        running = True
        while running:
            dt = clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_1:
                        self.belt_speed = 1
                    elif event.key == pygame.K_2:
                        self.belt_speed = 2
                    elif event.key == pygame.K_3:
                        self.belt_speed = 3

            if not self.paused:
                self._update(dt)

            self._drain_results()
            self._draw(screen)
            pygame.display.flip()

            if self.img_index >= len(self.images) and not self.items:
                time.sleep(1.5)
                running = False

        pygame.quit()

        self.results = {
            "total":       self.total,
            "correct":     self.correct,
            "score":       self.score,
            "best_streak": self.best_streak,
            "per_class":   {k: tuple(v) for k, v in self.per_class.items()},
        }
        return self.results

    # ── Update ───────────────────────────────────────────────────────────────

    def _update(self, dt):
        speed_px = self.belt_speed * 80 * dt
        self.belt_offset = (self.belt_offset + speed_px) % 40

        # Spawn next image
        if self.img_index < len(self.images):
            self.spawn_timer += dt
            gap = max(0.7, self.spawn_gap - self.belt_speed * 0.2)
            if self.spawn_timer >= gap:
                self._spawn_item()
                self.spawn_timer = 0.0

        # Move items
        for item in self.items:
            item.x -= speed_px
            item.age += dt
            item.flash_timer = max(0.0, item.flash_timer - dt)

            # Trigger classification when item reaches scan line
            if not item.classified and item.x <= self.SCAN_X:
                item.classified = True
                self._clf_queue.put(item)
                self.pipeline_log = ["Stage 1: frame captured", "Stage 2: preprocessing…"]

        # Remove items that have left the screen
        self.items = [i for i in self.items if i.x > -100]

        # Animate arm
        self.arm_angle += (self.arm_target - self.arm_angle) * min(1.0, dt * 6)

        # Decay feedback
        self.feedback = [
            (t, x, y, c, a - dt * 1.6, dy - dt * 30)
            for (t, x, y, c, a, dy) in self.feedback
            if a > 0
        ]

    def _spawn_item(self):
        idx   = self.img_index
        img   = self.images[idx]
        label = self.labels[idx]
        self.img_index += 1

        surf = self._make_surface(img)
        item = ConveyorItem(surf, label, get_bin(label), self.W + self.ITEM_SIZE)
        item.y = self.BELT_Y
        self.items.append(item)

    def _make_surface(self, np_image):
        """Convert a numpy HxWx3 uint8 array to a pygame Surface (scaled up)."""
        img = np.ascontiguousarray(np_image)
        surf = pygame.surfarray.make_surface(img.transpose(1, 0, 2))
        return pygame.transform.scale(surf, (self.ITEM_SIZE, self.ITEM_SIZE))

    # ── Classification worker thread ──────────────────────────────────────────

    def _classify_worker(self):
        while True:
            item = self._clf_queue.get()
            np_img = pygame.surfarray.array3d(item.surface).transpose(1, 0, 2)
            result = self.pipeline.process(np_img, item.true_label)
            self._res_queue.put((item, result))

    def _drain_results(self):
        while not self._res_queue.empty():
            item, result = self._res_queue.get_nowait()
            item.result = result

            self.total += 1
            cls = item.true_label
            self.per_class[cls][1] += 1

            if result["is_correct"]:
                self.correct += 1
                self.streak  += 1
                self.best_streak = max(self.best_streak, self.streak)
                self.score   += 10 + self.streak * 2
                self.per_class[cls][0] += 1
                pts = 10 + self.streak * 2
                self.feedback.append((f"+{pts}", item.x, item.y - 40, GREEN, 1.0, 0))
                item.flash_timer = 0.4
            else:
                self.streak = 0
                self.feedback.append(("ERR", item.x, item.y - 40, RED, 1.0, 0))
                item.flash_timer = -0.4    # negative = red flash

            # Point arm toward the target bin
            bins   = list(BIN_COLORS.keys())
            b_idx  = bins.index(result["predicted_bin"]) if result["predicted_bin"] in bins else 0
            angles = [-45, -15, 15, 45]
            self.arm_target = angles[b_idx]

            self.pipeline_log = result["stage_log"][-6:]

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self, screen):
        screen.fill(BG)

        self._draw_header(screen)
        self._draw_belt(screen)
        self._draw_items(screen)
        self._draw_scan_line(screen)
        self._draw_arm(screen)
        self._draw_bins(screen)
        self._draw_pipeline_log(screen)
        self._draw_stats(screen)
        self._draw_feedback(screen)
        self._draw_controls_hint(screen)

    def _draw_header(self, screen):
        title = FONT_LG.render("SortBot Vision System", True, WHITE)
        screen.blit(title, (20, 14))

        clf_txt = f"Classifier: {self.pipeline.classifier.upper()}  |  "
        clf_txt += f"Noise: {'OFF' if self.pipeline.noise_level == 0 else self.pipeline.noise_level}  |  "
        clf_txt += f"Effective acc: {self.pipeline.effective_accuracy()*100:.0f}%  |  "
        clf_txt += f"Avg latency: {self.pipeline.avg_latency():.0f}ms"
        sub = FONT_SM.render(clf_txt, True, MUTED)
        screen.blit(sub, (20, 40))

        if self.paused:
            p = FONT_LG.render("PAUSED", True, AMBER)
            screen.blit(p, (self.W // 2 - 40, 14))

    def _draw_belt(self, screen):
        # Belt body
        belt_rect = pygame.Rect(0, self.BELT_Y - self.BELT_H//2, self.W, self.BELT_H)
        pygame.draw.rect(screen, BELT_DARK, belt_rect)

        # Animated stripes
        stripe_x = -self.belt_offset
        while stripe_x < self.W:
            pygame.draw.line(screen, BELT_STRIPE,
                             (int(stripe_x), self.BELT_Y - self.BELT_H//2),
                             (int(stripe_x), self.BELT_Y + self.BELT_H//2), 1)
            stripe_x += 40

        # Belt edges
        pygame.draw.line(screen, MUTED, (0, self.BELT_Y - self.BELT_H//2),
                         (self.W, self.BELT_Y - self.BELT_H//2), 2)
        pygame.draw.line(screen, MUTED, (0, self.BELT_Y + self.BELT_H//2),
                         (self.W, self.BELT_Y + self.BELT_H//2), 2)

    def _draw_items(self, screen):
        for item in self.items:
            x = int(item.x)
            y = int(item.y)
            hw = self.ITEM_SIZE // 2

            # Flash on sort
            if item.flash_timer > 0:
                flash = pygame.Surface((self.ITEM_SIZE + 6, self.ITEM_SIZE + 6), pygame.SRCALPHA)
                alpha = int(min(255, item.flash_timer * 600))
                flash.fill((*GREEN, alpha))
                screen.blit(flash, (x - hw - 3, y - hw - 3))
            elif item.flash_timer < 0:
                flash = pygame.Surface((self.ITEM_SIZE + 6, self.ITEM_SIZE + 6), pygame.SRCALPHA)
                alpha = int(min(255, abs(item.flash_timer) * 600))
                flash.fill((*RED, alpha))
                screen.blit(flash, (x - hw - 3, y - hw - 3))

            screen.blit(item.surface, (x - hw, y - hw))

            # Border
            col = (50, 50, 55)
            if item.result:
                col = GREEN if item.result["is_correct"] else RED
            pygame.draw.rect(screen, col, (x - hw, y - hw, self.ITEM_SIZE, self.ITEM_SIZE), 1)

            # Label above
            if item.classified and item.result:
                pred = item.result["prediction"]
                conf = item.result["confidence"]
                lbl  = FONT_SM.render(f"{pred} {conf:.0%}", True, WHITE)
                screen.blit(lbl, (x - lbl.get_width()//2, y - hw - 16))

    def _draw_scan_line(self, screen):
        pulse = 0.6 + 0.4 * math.sin(time.time() * 4)
        col   = tuple(int(c * pulse) for c in GREEN)
        pygame.draw.line(screen, col,
                         (self.SCAN_X, self.BELT_Y - self.BELT_H//2 - 10),
                         (self.SCAN_X, self.BELT_Y + self.BELT_H//2 + 10), 2)
        lbl = FONT_SM.render("SCAN", True, col)
        screen.blit(lbl, (self.SCAN_X - lbl.get_width()//2,
                          self.BELT_Y - self.BELT_H//2 - 26))

    def _draw_arm(self, screen):
        """Draw the sorting actuator arm pivoting from a point above the bins."""
        pivot_x = self.SCAN_X + 60
        pivot_y = self.BELT_Y + self.BELT_H//2 + 10
        arm_len = 55
        angle_r = math.radians(self.arm_angle)
        end_x = pivot_x + arm_len * math.sin(angle_r)
        end_y = pivot_y + arm_len * math.cos(angle_r)
        pygame.draw.line(screen, AMBER, (pivot_x, pivot_y), (int(end_x), int(end_y)), 3)
        pygame.draw.circle(screen, AMBER, (pivot_x, pivot_y), 5)

    def _draw_bins(self, screen):
        bins    = list(BIN_COLORS.keys())
        n       = len(bins)
        spacing = self.W // n
        for i, bin_name in enumerate(bins):
            cx = spacing // 2 + i * spacing
            cy = self.BIN_Y
            col = BIN_COLORS[bin_name]
            r   = pygame.Rect(cx - self.BIN_W//2, cy - self.BIN_H//2,
                              self.BIN_W, self.BIN_H)
            pygame.draw.rect(screen, (col[0]//5, col[1]//5, col[2]//5), r, border_radius=8)
            pygame.draw.rect(screen, col, r, width=2, border_radius=8)

            lbl  = FONT_SANS.render(BIN_LABELS[bin_name], True, col)
            screen.blit(lbl, (cx - lbl.get_width()//2, cy - 14))

            # Count
            cnt_correct = self.per_class  # we'll count from per_class
            # count items in this bin
            bin_total = sum(
                v[0] for k, v in self.per_class.items()
                if get_bin(k) == bin_name
            )
            cnt = FONT_LG.render(str(bin_total), True, WHITE)
            screen.blit(cnt, (cx - cnt.get_width()//2, cy + 4))

    def _draw_pipeline_log(self, screen):
        panel = pygame.Rect(20, 340, 340, 110)
        pygame.draw.rect(screen, (28, 28, 32), panel, border_radius=6)
        pygame.draw.rect(screen, (50, 50, 55), panel, width=1, border_radius=6)

        hdr = FONT_SM.render("Vision pipeline log", True, MUTED)
        screen.blit(hdr, (30, 348))

        for i, line in enumerate(self.pipeline_log[-5:]):
            col  = GREEN if i == len(self.pipeline_log[-5:]) - 1 else MUTED
            surf = FONT_MONO.render(line[:48], True, col)
            screen.blit(surf, (30, 364 + i * 17))

    def _draw_stats(self, screen):
        panel = pygame.Rect(380, 340, 360, 110)
        pygame.draw.rect(screen, (28, 28, 32), panel, border_radius=6)
        pygame.draw.rect(screen, (50, 50, 55), panel, width=1, border_radius=6)

        acc = self.correct / max(self.total, 1) * 100
        stats = [
            ("Score",    str(self.score),           WHITE),
            ("Accuracy", f"{acc:.1f}%",             GREEN if acc > 75 else AMBER if acc > 55 else RED),
            ("Streak",   str(self.streak),          AMBER),
            ("Sorted",   f"{self.total}/{len(self.images)}", MUTED),
        ]
        for i, (lbl, val, col) in enumerate(stats):
            x = 390 + (i % 2) * 170
            y = 350 + (i // 2) * 48
            l = FONT_SM.render(lbl, True, MUTED)
            v = FONT_LG.render(val, True, col)
            screen.blit(l, (x, y))
            screen.blit(v, (x, y + 16))

    def _draw_feedback(self, screen):
        for (txt, x, y, col, alpha, dy) in self.feedback:
            if alpha <= 0:
                continue
            surf = FONT_LG.render(txt, True, col)
            surf.set_alpha(int(alpha * 255))
            screen.blit(surf, (int(x) - surf.get_width()//2, int(y + dy)))

    def _draw_controls_hint(self, screen):
        hint = "SPACE pause  |  1/2/3 speed  |  Q quit"
        s = FONT_SM.render(hint, True, (60, 60, 65))
        screen.blit(s, (self.W - s.get_width() - 10, self.H - 20))
