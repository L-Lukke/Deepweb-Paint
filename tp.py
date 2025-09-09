import tkinter as tk
from tkinter import ttk, messagebox
import math
from dataclasses import dataclass
from typing import List, Tuple

# conversor básico de graus -> radianos
def rad(deg: float) -> float:
    return deg * math.pi / 180.0

def mat_mul(A, B):
    out = [[0]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            out[i][j] = sum(A[i][k]*B[k][j] for k in range(3))
    return out

def mat_vec(M, v):
    x,y = v
    hv = [x, y, 1]
    res = [sum(M[i][k]*hv[k] for k in range(3)) for i in range(3)]
    return (res[0], res[1])

# transformações básicas
def T(dx, dy): return [[1,0,dx],[0,1,dy],[0,0,1]]
def S(sx, sy): return [[sx,0,0],[0,sy,0],[0,0,1]]
def R(deg):
    c = math.cos(rad(deg)); s = math.sin(rad(deg))
    return [[c,-s,0],[s,c,0],[0,0,1]]
def reflect_x(): return [[1,0,0],[0,-1,0],[0,0,1]]
def reflect_y(): return [[-1,0,0],[0,1,0],[0,0,1]]

# ---- estruturas geométricas básicas ----

@dataclass
class Vertex:
    x: int; y: int
    def transform(self, M):
        x,y = mat_vec(M,(self.x,self.y))
        return Vertex(int(round(x)), int(round(y)))

@dataclass
class Line:
    a: Vertex; b: Vertex; selected: bool = False
    def vertices(self): return [self.a, self.b]
    def transform(self, M): return Line(self.a.transform(M), self.b.transform(M), self.selected)

@dataclass
class Circle:
    center: Vertex
    radius: int
    selected: bool = False

    def vertices(self):
        return [self.center]

    def transform(self, M):
        new_center = self.center.transform(M)
        unit_x = Vertex(1, 0).transform(M)
        scale_x = math.dist((0, 0), (unit_x.x, unit_x.y))
        new_radius = max(1, int(round(self.radius * scale_x))) 

        return Circle(new_center, new_radius, self.selected)

@dataclass
class Polygon:
    vertices_list: List[Vertex]; selected: bool = False
    def vertices(self): return self.vertices_list
    def transform(self, M): return Polygon([v.transform(M) for v in self.vertices_list], self.selected)

@dataclass
class Point:
    p: Vertex; selected: bool = False
    def vertices(self): return [self.p]
    def transform(self, M): return Point(self.p.transform(M), self.selected)

# ---- rasterização de retas ----

def dda_line(x0,y0,x1,y1):
    pts = []
    dx, dy = x1-x0, y1-y0
    steps = int(max(abs(dx), abs(dy)))
    if steps == 0: return [(x0,y0)]
    x_inc, y_inc = dx/steps, dy/steps
    x, y = x0, y0
    for _ in range(steps+1):
        pts.append((int(round(x)), int(round(y))))
        x += x_inc; y += y_inc
    return pts

def bresenham_line(x0,y0,x1,y1):
    pts = []
    dx = abs(x1-x0); sx = 1 if x0 < x1 else -1
    dy = -abs(y1-y0); sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        pts.append((x0,y0))
        if x0 == x1 and y0 == y1: break
        e2 = 2*err
        if e2 >= dy: err += dy; x0 += sx
        if e2 <= dx: err += dx; y0 += sy
    return pts

# ---- rasterização de circunferência ----

def bresenham_circle(xc: int, yc: int, r: int):
    # algoritmo do ponto médio: desenha 1/8 e espelha 8 vezes
    pts = []
    x, y = 0, r
    d = 3 - 2 * r

    def plot8(cx, cy, px, py):
        pts.append((cx + px, cy + py))
        pts.append((cx - px, cy + py))
        pts.append((cx + px, cy - py))
        pts.append((cx - px, cy - py))
        pts.append((cx + py, cy + px))
        pts.append((cx - py, cy + px))
        pts.append((cx + py, cy - px))
        pts.append((cx - py, cy - px))

    while y >= x:
        plot8(xc, yc, x, y)
        if d <= 0:
            d = d + 4 * x + 6
        else:
            d = d + 4 * (x - y) + 10
            y -= 1
        x += 1

    pts = list(dict.fromkeys(pts))
    return pts

# ---- recorte de linhas ----

INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

def _compute_out_code(x,y,xmin,ymin,xmax,ymax):
    code = INSIDE
    if x < xmin: code |= LEFT
    elif x > xmax: code |= RIGHT
    if y < ymin: code |= BOTTOM
    elif y > ymax: code |= TOP
    return code

def cohen_sutherland_clip(x0,y0,x1,y1,xmin,ymin,xmax,ymax):
    out0 = _compute_out_code(x0,y0,xmin,ymin,xmax,ymax)
    out1 = _compute_out_code(x1,y1,xmin,ymin,xmax,ymax)
    while True:
        if not (out0 | out1):
            return True, (x0,y0,x1,y1)       # trivialmente aceito
        if out0 & out1:
            return False, (x0,y0,x1,y1)      # trivialemnte rejeitado
        out_out = out0 if out0 else out1     # pega um endpoint fora
        # calcula interseção com a borda adequada
        if out_out & TOP:
            x = x0 + (x1-x0) * (ymax - y0) / (y1 - y0) if (y1-y0)!=0 else x0
            y = ymax
        elif out_out & BOTTOM:
            x = x0 + (x1-x0) * (ymin - y0) / (y1 - y0) if (y1-y0)!=0 else x0
            y = ymin
        elif out_out & RIGHT:
            y = y0 + (y1-y0) * (xmax - x0) / (x1 - x0) if (x1-x0)!=0 else y0
            x = xmax
        else:
            y = y0 + (y1-y0) * (xmin - x0) / (x1 - x0) if (x1-x0)!=0 else y0
            x = xmin
        if out_out == out0:
            x0, y0 = int(round(x)), int(round(y)); out0 = _compute_out_code(x0,y0,xmin,ymin,xmax,ymax)
        else:
            x1, y1 = int(round(x)), int(round(y)); out1 = _compute_out_code(x1,y1,xmin,ymin,xmax,ymax)

def liang_barsky_clip(x0,y0,x1,y1,xmin,ymin,xmax,ymax):
    dx, dy = x1 - x0, y1 - y0
    p = [-dx, dx, -dy, dy]
    q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False, (x0, y0, x1, y1)  # paralelo fora
            continue
        r = qi / pi
        if pi < 0:
            if r > u2: return False, (x0, y0, x1, y1)
            if r > u1: u1 = r
        else:
            if r < u1: return False, (x0, y0, x1, y1)
            if r < u2: u2 = r
    nx0 = int(round(x0 + u1 * dx)); ny0 = int(round(y0 + u1 * dy))
    nx1 = int(round(x0 + u2 * dx)); ny1 = int(round(y0 + u2 * dy))
    return True, (nx0, ny0, nx1, ny1)

# ---- Canvas baseado em pixels com origem no centro ----

class PixelCanvas(tk.Frame):
    def __init__(self, master, width=1200, height=800, **kwargs):
        super().__init__(master, **kwargs)
        self.width = width; self.height = height
        self.tkcanvas = tk.Canvas(self, width=width, height=height, bg="white", highlightthickness=0)
        self.tkcanvas.pack()
        self.img = tk.PhotoImage(width=width, height=height)
        self.image_id = self.tkcanvas.create_image(0, 0, anchor="nw", image=self.img)

    def world_to_screen(self, x: int, y: int) -> Tuple[int,int]:
        sx = int(round(x + self.width/2))
        sy = int(round(self.height/2 - y))
        return sx, sy

    def screen_to_world(self, sx: int, sy: int) -> Tuple[int,int]:
        x = int(round(sx - self.width/2))
        y = int(round(self.height/2 - sy))
        return x, y

    def clear(self):
        self.img.put(("white",), to=(0,0,self.width,self.height))
        self.tkcanvas.delete("overlay")

    def set_pixel_screen(self, sx, sy, color="black"):
        if 0 <= sx < self.width and 0 <= sy < self.height:
            self.img.put(color, (sx, sy))

    def set_pixel_world(self, x, y, color="black"):
        sx, sy = self.world_to_screen(x, y)
        self.set_pixel_screen(sx, sy, color)

    def draw_point_world(self, x, y, color="black"):
        self.set_pixel_world(x, y, color)

    def draw_line_pixels_world(self, world_points, color="black"):
        for (x,y) in world_points:
            self.set_pixel_world(x,y,color)

    def draw_rect_outline_screen(self, sx0,sy0,sx1,sy1, color="blue", dash=(3,2), tag="overlay"):
        self.tkcanvas.delete(tag)
        self.tkcanvas.create_rectangle(sx0,sy0,sx1,sy1, outline=color, dash=dash, width=1, tags=("overlay", tag))

# ---- App principal (UI lateral + interação + render loop) ----

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CG TP")
        self.geometry("1520x880"); self.resizable(False, False)

        # área de desenho
        self.canvas = PixelCanvas(self, width=1200, height=800)
        self.canvas.place(x=280, y=20)

        # variáveis de estado
        self.mode = tk.StringVar(value="select")
        self.raster = tk.StringVar(value="bresenham")
        self.clip_algo = tk.StringVar(value="cohen")
        self.tmp_points: List[Vertex] = []   # buffer de cliques temporários
        self.objects: List[object] = []      # cena
        self.selected_rect_world = None
        self.clip_rect_world = None
        self.drag_start_screen = None; self.dragging = False

        self._build_sidebar(); self._connect_events(); self.canvas.clear()
        self.redraw()

    def _build_sidebar(self):
        side = ttk.Frame(self, padding=8); side.place(x=10, y=10, width=260, height=860)

        ttk.Label(side, text="Modo de Edição").pack(anchor="w")
        for text, val in [("Select / Box","select"),
                  ("Add Point","add_point"),
                  ("Add Line","add_line"),
                  ("Add Polygon","add_polygon"),
                  ("Add Circle","add_circle"),
                  ("Set Clipping Window","clip_window")]:
            ttk.Radiobutton(side, text=text, variable=self.mode, value=val).pack(anchor="w")

        ttk.Separator(side).pack(fill="x", pady=6)
        ttk.Label(side, text="Line Rasterization").pack(anchor="w")
        ttk.Radiobutton(side, text="Bresenham", variable=self.raster, value="bresenham").pack(anchor="w")
        ttk.Radiobutton(side, text="DDA", variable=self.raster, value="dda").pack(anchor="w")

        ttk.Separator(side).pack(fill="x", pady=6)
        ttk.Label(side, text="Clipping").pack(anchor="w")
        ttk.Radiobutton(side, text="Cohen–Sutherland", variable=self.clip_algo, value="cohen").pack(anchor="w")
        ttk.Radiobutton(side, text="Liang–Barsky", variable=self.clip_algo, value="liang").pack(anchor="w")
        ttk.Button(side, text="Apply Clipping to Selected", command=self.apply_clip_selected).pack(anchor="w", pady=4)

        ttk.Separator(side).pack(fill="x", pady=6)
        tf = ttk.LabelFrame(side, text="2D Transformations (on selected)"); tf.pack(fill="x", pady=4)

        # translação
        row = ttk.Frame(tf); row.pack(fill="x", pady=2)
        ttk.Label(row, text="dx:").pack(side="left"); self.dx = tk.DoubleVar(value=0.0)
        ttk.Entry(row, textvariable=self.dx, width=6).pack(side="left", padx=4)
        ttk.Label(row, text="dy:").pack(side="left"); self.dy = tk.DoubleVar(value=0.0)
        ttk.Entry(row, textvariable=self.dy, width=6).pack(side="left", padx=4)
        ttk.Button(tf, text="Translate", command=self.apply_translate).pack(fill="x", pady=2)

        # escala
        row = ttk.Frame(tf); row.pack(fill="x", pady=2)
        ttk.Label(row, text="sx:").pack(side="left"); self.sx = tk.DoubleVar(value=1.0)
        ttk.Entry(row, textvariable=self.sx, width=6).pack(side="left", padx=4)
        ttk.Label(row, text="sy:").pack(side="left"); self.sy = tk.DoubleVar(value=1.0)
        ttk.Entry(row, textvariable=self.sy, width=6).pack(side="left", padx=4)
        ttk.Button(tf, text="Scale", command=self.apply_scale).pack(fill="x", pady=2)

        # rotação
        row = ttk.Frame(tf); row.pack(fill="x", pady=2)
        ttk.Label(row, text="θ:").pack(side="left"); self.angle = tk.DoubleVar(value=0.0)
        ttk.Entry(row, textvariable=self.angle, width=6).pack(side="left", padx=4)
        ttk.Button(tf, text="Rotate", command=self.apply_rotate).pack(fill="x", pady=2)

        # reflexões
        ttk.Label(tf, text="Reflexões (eixos)").pack(anchor="w", pady=(6,0))
        rrow = ttk.Frame(tf); rrow.pack(fill="x", pady=2)
        ttk.Button(rrow, text="Reflect X", command=self.apply_reflect_x).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(rrow, text="Reflect Y", command=self.apply_reflect_y).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(tf, text="Reflext XY", command=self.apply_reflect_xy).pack(fill="x", pady=2)

        # utilidades
        ttk.Separator(side).pack(fill="x", pady=6)
        ttk.Button(side, text="Clear", command=self.clear_all).pack(fill="x")
        ttk.Label(side, text="Click to create; drag to select.\nDouble-click closes polygon.", wraplength=230).pack(anchor="w", pady=6)
        ttk.Button(side, text="Close Current Polygon", command=self.finish_polygon).pack(fill="x")

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(side, textvariable=self.status, relief="sunken").pack(fill="x", pady=(8,0))

    def _connect_events(self):
        cv = self.canvas.tkcanvas
        cv.bind("<ButtonPress-1>", self.on_mouse_down)
        cv.bind("<B1-Motion>", self.on_mouse_drag)
        cv.bind("<ButtonRelease-1>", self.on_mouse_up)
        cv.bind("<Double-Button-1>", self.on_double_click)
        cv.bind("<Button-3>", lambda e: "break") # desabilita menu contexto

    def _current_raster_fn(self):
        return bresenham_line if self.raster.get()=="bresenham" else dda_line

    def on_mouse_down(self, event):
        m = self.mode.get()
        if m in ("select","clip_window"):
            self.drag_start_screen = (event.x, event.y); self.dragging = True
        elif m == "add_point":
            x,y = self.canvas.screen_to_world(event.x, event.y)
            self.objects.append(Point(Vertex(x,y)))
            self.status.set(f"Point in ({x},{y})."); self.redraw()
        elif m == "add_line":
            x,y = self.canvas.screen_to_world(event.x, event.y)
            self.tmp_points.append(Vertex(x,y))
            if len(self.tmp_points)==2:
                a,b = self.tmp_points; self.objects.append(Line(a,b)); self.tmp_points.clear()
                self.status.set(f"Line in ({a.x},{a.y})-({b.x},{b.y})."); self.redraw()
        elif m == "add_polygon":
            x,y = self.canvas.screen_to_world(event.x, event.y)
            self.tmp_points.append(Vertex(x,y))
            self.status.set(f"Vertex {len(self.tmp_points)} in ({x},{y})."); self.redraw()
        elif m == "add_circle":
            # 1º clique = centro; 2º clique = ponto na borda
            x, y = self.canvas.screen_to_world(event.x, event.y)
            self.tmp_points.append(Vertex(x, y))
            if len(self.tmp_points) == 1:
                self.status.set(f"Circle center: ({x},{y}). Click again to define radius.")
            elif len(self.tmp_points) == 2:
                c = self.tmp_points[0]
                p = self.tmp_points[1]
                r = int(round(math.dist((c.x, c.y), (p.x, p.y))))
                r = max(1, r)
                self.objects.append(Circle(c, r))
                self.tmp_points.clear()
                self.status.set(f"Circle created. Center=({c.x},{c.y}), radius={r}.")
                self.redraw()

    def on_double_click(self, event):
        # double-click fecha polígono
        if len(self.tmp_points) >= 3:
            self.finish_polygon()

    def on_mouse_drag(self, event):
        # feedback visual de seleção
        m = self.mode.get()
        if m in ("select","clip_window") and self.dragging:
            x0,y0 = self.drag_start_screen; x1,y1 = event.x, event.y
            tag = "select_rect" if m=="select" else "clip_rect"
            color = "green" if m=="select" else "red"
            self.canvas.draw_rect_outline_screen(x0,y0,x1,y1,color=color, tag=tag)

    def on_mouse_up(self, event):
        # soltar o mouse commita a seleção
        m = self.mode.get()
        if m in ("select","clip_window") and self.dragging:
            self.dragging = False
            sx0,sy0 = self.drag_start_screen; sx1,sy1 = event.x, event.y
            x0,x1 = sorted((sx0,sx1)); y0,y1 = sorted((sy0,sy1))
            wx0,wy0 = self.canvas.screen_to_world(x0,y1)
            wx1,wy1 = self.canvas.screen_to_world(x1,y0)
            xmin,xmax = sorted((wx0,wx1)); ymin,ymax = sorted((wy0,wy1))
            if m == "select":
                # marca objetos cujo bbox intersecta a seleção
                self.selected_rect_world = (xmin,ymin,xmax,ymax)
                self.select_in_rect_world(xmin,ymin,xmax,ymax)
                self.status.set(f"Selected in ({xmin},{ymin})-({xmax},{ymax}).")
                self.canvas.tkcanvas.delete("select_rect")
            else:
                self.clip_rect_world = (xmin,ymin,xmax,ymax)
                self.status.set(f"Clipping window in ({xmin},{ymin})-({xmax},{ymax}).")
            self.redraw()

    def bbox_of_object_world(self, obj) -> Tuple[int,int,int,int]:
        # bounding box genérico para seleção
        if isinstance(obj, Circle):
            cx, cy = obj.center.x, obj.center.y
            r = obj.radius
            return (cx - r, cy - r, cx + r, cy + r)
        xs = [v.x for v in obj.vertices()]; ys = [v.y for v in obj.vertices()]
        return (min(xs), min(ys), max(xs), max(ys))

    def select_in_rect_world(self, xmin,ymin,xmax,ymax):
        # marca como selected toda geometria com bbox intersectando o retângulo
        for obj in self.objects:
            bx0,by0,bx1,by1 = self.bbox_of_object_world(obj)
            obj.selected = not (bx1 < xmin or bx0 > xmax or by1 < ymin or by0 > ymax)

    def _apply_transform_selected(self, M):
        # aplica transformação apenas nos selecionados
        new_objs = []
        for obj in self.objects:
            new_objs.append(obj.transform(M) if getattr(obj,"selected",False) else obj)
        self.objects = new_objs; self.redraw()

    # botões de transformação mapeados direto
    def apply_translate(self): self._apply_transform_selected(T(self.dx.get(), self.dy.get()))
    def apply_scale(self): self._apply_transform_selected(S(self.sx.get(), self.sy.get()))
    def apply_rotate(self): self._apply_transform_selected(R(self.angle.get()))
    def apply_reflect_x(self): self._apply_transform_selected(reflect_x())
    def apply_reflect_y(self): self._apply_transform_selected(reflect_y())
    def apply_reflect_xy(self): self._apply_transform_selected(mat_mul(reflect_x(), reflect_y()))

    def apply_clip_selected(self):
        # recorta só as linhas selecionadas
        if not self.clip_rect_world:
            messagebox.showwarning("Recorte", "Defina a janela de recorte primeiro.")
            return
        xmin,ymin,xmax,ymax = self.clip_rect_world
        algo = self.clip_algo.get()
        new_objs = []
        for obj in self.objects:
            if isinstance(obj, Line) and obj.selected:
                x0,y0,x1,y1 = obj.a.x,obj.a.y,obj.b.x,obj.b.y
                if algo == "cohen":
                    ok, (cx0,cy0,cx1,cy1) = cohen_sutherland_clip(x0,y0,x1,y1,xmin,ymin,xmax,ymax)
                else:
                    ok, (cx0,cy0,cx1,cy1) = liang_barsky_clip(x0,y0,x1,y1,xmin,ymin,xmax,ymax)
                if ok: new_objs.append(Line(Vertex(cx0,cy0), Vertex(cx1,cy1), selected=True))
            else:
                new_objs.append(obj)
        self.objects = new_objs
        self.status.set(f"Clipping '{algo}' applied."); self.redraw()

    def draw_axes(self, raster_fn):
        # desenha os eixos x e y com o raster escolhido
        W,H = self.canvas.width, self.canvas.height
        half_w = int(W/2); half_h = int(H/2)
        pts = raster_fn(-half_w, 0, half_w, 0)
        self.canvas.draw_line_pixels_world(pts, color="gray75")
        pts = raster_fn(0, -half_h, 0, half_h)
        self.canvas.draw_line_pixels_world(pts, color="gray75")

    def redraw(self):
        self.canvas.clear()
        raster = self._current_raster_fn()
        self.draw_axes(raster)

        # mostra janela de recorte, se existir
        if self.clip_rect_world:
            xmin,ymin,xmax,ymax = self.clip_rect_world
            sx0,sy0 = self.canvas.world_to_screen(xmin,ymin)
            sx1,sy1 = self.canvas.world_to_screen(xmax,ymax)
            x0,x1 = sorted((sx0,sx1)); y0,y1 = sorted((sy0,sy1))
            self.canvas.draw_rect_outline_screen(x0,y0,x1,y1, color="red", tag="clip_rect")

        # desenha a cena
        for obj in self.objects:
            color = "black" if not getattr(obj,"selected",False) else "blue"
            if isinstance(obj, Point):
                self.canvas.draw_point_world(obj.p.x, obj.p.y, color)
            elif isinstance(obj, Line):
                pts = raster(obj.a.x, obj.a.y, obj.b.x, obj.b.y)
                self.canvas.draw_line_pixels_world(pts, color)
            elif isinstance(obj, Polygon):
                vs = obj.vertices_list
                for i in range(len(vs)):
                    a = vs[i]; b = vs[(i+1)%len(vs)]
                    pts = raster(a.x, a.y, b.x, b.y)
                    self.canvas.draw_line_pixels_world(pts, color)
            elif isinstance(obj, Circle):
                pts = bresenham_circle(obj.center.x, obj.center.y, obj.radius)
                self.canvas.draw_line_pixels_world(pts, color)

        # feedback visual para objetos em construção
        if len(self.tmp_points) > 0:
            for v in self.tmp_points:
                self.canvas.draw_point_world(v.x, v.y, "green")
            for i in range(len(self.tmp_points)-1):
                a = self.tmp_points[i]; b = self.tmp_points[i+1]
                pts = raster(a.x,a.y,b.x,b.y)
                self.canvas.draw_line_pixels_world(pts, "green")
            if len(self.tmp_points) >= 2:
                a = self.tmp_points[-1]; b = self.tmp_points[0]
                pts = raster(a.x,a.y,b.x,b.y)
                self.canvas.draw_line_pixels_world(pts, "green")

    def finish_polygon(self):
        # fecha o polígono quando houver pelo menos 3 vértices
        if len(self.tmp_points) < 3:
            messagebox.showwarning("Polygon", "A polygon needs at least 3 vertexes.")
            return
        self.objects.append(Polygon(self.tmp_points.copy()))
        self.tmp_points.clear()
        self.status.set("Polygon closed."); self.redraw()

    def clear_all(self):
        # reset da cena/estado
        self.objects.clear(); self.tmp_points.clear()
        self.selected_rect_world = None; self.clip_rect_world = None
        self.canvas.clear(); self.redraw()

# entry point
def main():
    app = App(); app.mainloop()

if __name__ == "__main__":
    main()
