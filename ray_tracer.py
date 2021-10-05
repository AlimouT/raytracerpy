from abc import abstractmethod
from math import sqrt
import cv2
import numpy as np


### Coordinates - 3D
#   x: left to right
#   y: up to down, as is convention with open-cv images, which start a the top left
#   z: behind to forward

### BASE MATH

class Pt:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def distance(pt1, pt2):
        return sqrt(
            (pt2.x - pt1.x)**2 +
            (pt2.y - pt1.y)**2 +
            (pt2.z - pt1.x)**2
        )

    def __add__(self, other):
        return Pt(
           self.x + other.x,
           self.y + other.y,
           self.z + other.z
        )

    def __sub__(self, other):
        return Pt(
           self.x - other.x,
           self.y - other.y,
           self.z - other.z
        )

    def __repr__(self):
        return f'Pt({self.x},{self.y},{self.z})'


class Vec:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def unit_vec(self):
        return UVec(self.x, self.y, self.z)

    @staticmethod
    def between(pt1:Pt, pt2:Pt):
        return Vec(
            pt2.x - pt1.x,
            pt2.y - pt1.y,
            pt2.z - pt1.z,
        )

    def __mul__(self, scalar):
        return Vec(
            self.x*scalar,
            self.y*scalar,
            self.z*scalar
        )


class UVec(Vec):
#   Unit Vector
    def __init__(self, x, y, z):
        length = sqrt(x**2 + y**2 + z**2)
        try:
            ux = x/length
            uy = y/length
            uz = z/length
        except ZeroDivisionError as err:
            print(f'({x},{y},{z}) is not a valid unit vector.')
            raise Exception('Invalid UnitVector') from err
        super().__init__(ux,uy,uz)

    @staticmethod
    def between(pt1:Pt, pt2:Pt):
        return UVec(
            pt2.x - pt1.x,
            pt2.y - pt1.y,
            pt2.z - pt1.z,
        )

class Line:
    def __init__(self, origin:Pt, direction:Vec):
        self.origin = origin
        self.direction = direction


class Shape():
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def intersect(self, ray):
        pass


class Sphere(Shape):
    def __init__(self, radius, pos):
        self.rad = radius
        self.pos = pos

    def intersect(self, ray):
        ### Return the point of intersection between the object and the ray, if it exists.
        ray_ori = ray.origin
        ray_dir = ray.direction
        sphere_center = self.pos
        sphere_radius = self.rad

        det = (ray_dir.dot(ray_ori-sphere_center))**2 - (Pt.distance(ray_ori, sphere_center)**2 - sphere_radius**2)
        if det < 0:
            return False
        if det == 0:
            dis = -(ray_dir.dot(ray_ori-sphere_center))
        dis1 = -(ray_dir.dot(ray_ori-sphere_center)) - sqrt(det)
        dis2 = -(ray_dir.dot(ray_ori-sphere_center)) + sqrt(det)
        if abs(dis1) < abs(dis2):
            dis = dis1
        else:
            dis = dis2
        return ray_ori + ray_dir*dis


### Base Graphics

class Ray(Line):
    def __init__(self, start_pos, direction):
        super().__init__(start_pos, direction.unit_vec())


class Item():
    def __init__(self, shape:Shape, colour):
        self.shape = shape
        self.colour = colour

    def intersect(self, ray):
        return self.shape.intersect(ray)


class Screen:
    def __init__(self, width, height, eye:Pt, pos=None):
        self.width = width
        self.height = height
        self.eye = eye
        if pos is None:
            self.pos = Pt(-width/2, -height/2)
        else:
            self.pos = pos

class Scene:
    def __init__(self, light_source:Pt, brightness=1, items=None):
        self.light_source = light_source
        self.brightness = 1
        self.items = []

    def render(self, screen):
        ### Careful: Because of np's axis ordering, the coordinates and image axis are different
        image = np.zeros((screen.width, screen.height, 3), order='F')
        for x in range(screen.width):
            for y in range(screen.height):
                # Create eye ray; What can we see? Origin is the exit of the screen, direction is the unit vector of the vector between the eye and that point.
                ray_origin = screen.pos + Pt(x,y)
                vision_ray = Ray(ray_origin, UVec.between(screen.eye, ray_origin))
                collision_pt = None
                min_distance = float('inf')
                for item in self.items:
                    pt_hit = item.intersect(vision_ray)
                    if pt_hit:
                        hit_distance = Pt.distance(ray_origin, pt_hit)
                        if hit_distance < min_distance:
                            min_distance = hit_distance
                            collision_pt = pt_hit
                            collision_item = item

                if collision_pt is None:
                    # No item to see.
                    continue

                # Create shadow ray; Is this position illuminated?
                shadow_ray = Ray(collision_pt, Vec.between(collision_pt, self.light_source))
                shadowed = False
                for item in self.items:
                    shadowed = item.intersect(shadow_ray)
                    if shadowed: break

                if shadowed:
                    image[x,y,:] = collision_item.colour *.6
                else:
                    image[x,y,:] = collision_item.colour * self.brightness
        return image

def test():
    scene = Scene(light_source=Pt(0,-1000,0))
    sphere1 = Item(Sphere(80, Pt(10, -100, 100)), np.array([1,0,0])) # B
    sphere2 = Item(Sphere(80, Pt(100, 100, 10)), np.array([0,1,0])) # G
    sphere3 = Item(Sphere(80, Pt(-70, 10, 100)), np.array([0,0,1])) # R
    scene.items.append(sphere1)
    scene.items.append(sphere2)
    scene.items.append(sphere3)
    screen = Screen(400, 600, eye=Pt(0,0,-1000))
    image = scene.render(screen)
    print(image.shape)
    cv2.imshow('image_render.jpg', image)
    cv2.waitKey()


def main():
    test()

if __name__ ==  "__main__":
    main()
