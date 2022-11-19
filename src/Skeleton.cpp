//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!!
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Dancs Bal�zs
// Neptun : AXDRVK
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const float epsilon = 0.0001f;
float fov = 45 * M_PI / 180;
vec3 eye = vec3(0, 0, 2), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);

GPUProgram gpuProgram;

const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord);
	}
)";

struct Material {
    vec3 ka, kd, ks;
    float  shininess;
    Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd * M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

struct Ray {
    vec3 start, dir;
    Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

struct Hit {
    float t;
    vec3 position, normal;
    Material * material;
    Hit() { t = -1; }
};

struct Light {
    vec3 direction;
    vec3 Le;
    Light(vec3 _direction, vec3 _Le) {
        direction = normalize(_direction);
        Le = _Le;
    }
};

class Camera {
    vec3 eye, lookat, right, up;
public:
    void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
        eye = _eye;
        lookat = _lookat;
        vec3 w = eye - lookat;
        float focus = length(w);
        right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
        up = normalize(cross(w, right)) * focus * tanf(fov / 2);
    }

    Ray getRay(int X, int Y) {
        vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
        return Ray(eye, dir);
    }
};

Camera camera;

class Intersectable {
protected:
    Material * material;
public:
    virtual Hit intersect(const Ray& ray) = 0;
};

struct Sphere : public Intersectable {
    vec3 center;
    float radius;

    Sphere(const vec3& _center, float _radius, Material* _material) {
        center = _center;
        radius = _radius;
        material = _material;
    }

    Hit intersect(const Ray& ray) {
        Hit hit;
        vec3 dist = ray.start - center;
        float a = dot(ray.dir, ray.dir);
        float b = dot(dist, ray.dir) * 2.0f;
        float c = dot(dist, dist) - radius * radius;
        float discr = b * b - 4.0f * a * c;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
        float t2 = (-b - sqrt_discr) / 2.0f / a;
        if (t1 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = (hit.position - center) * (1.0f / radius);
        hit.material = material;
        return hit;
    }
};

struct Triangle : public Intersectable {
    vec3 r1, r2 ,r3;

    Triangle(const vec3& r1, const vec3& r2, const vec3& r3, Material* material) {
        this->r1 = r1;
        this->r2 = r2;
        this->r3 = r3;
        this->material = material;
    }

    Hit intersect(const Ray& ray) {
        Hit hit;
        hit.normal = cross((r2 - r1), (r3 - r1));
        //printf("normal vector: x: %f, y: %f, z: %f", hit.normal.x, hit.normal.y, hit.normal.z);
        hit.t = dot((r1 - ray.start), hit.normal) / dot(ray.dir, hit.normal);
        hit.position = ray.start + ray.dir * hit.t;
        if (dot(cross((r2 - r1), (hit.position - r1)), hit.normal) <= 0.0f) hit.t = -1.0f;
        if (dot(cross((r3 - r2), (hit.position - r2)), hit.normal) <= 0.0f) hit.t = -1.0f;
        if (dot(cross((r1 - r3), (hit.position - r3)), hit.normal) <= 0.0f) hit.t = -1.0f;
        hit.material = material;
        return hit;
    }
};

float rnd() { return (float)rand() / RAND_MAX; }

class Scene {
    std::vector<Intersectable *> objects;
    std::vector<Light *> lights;
    vec3 La;
public:
    void build() {
        La = vec3(0.0f, 0.0f, 0.0f);
        vec3 lightDirection(1, 1, 1), Le(2, 2, 2);
        lights.push_back(new Light(lightDirection, Le));

        vec3 kd(0.3f, 0.0f, 0.0f), ks(2, 2, 2);
        Material * material = new Material(kd, ks, 50);
        for (int i = 0; i < 15; i++)
            objects.push_back(new Sphere(vec3(rnd() - 0.5f, rnd() - 0.5f, rnd() - 0.5f), rnd() * 0.1f, material));
        objects.push_back(new Triangle(vec3(-0.5f, -0.5f, 0.5f), vec3(0.5f, -0.5f, 0.5f), vec3(0.0f, 0.5f, 0.0f), material));
    }

    void render(std::vector<vec4>& image) {
        for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
            for (int X = 0; X < windowWidth; X++) {
                vec3 color = trace(camera.getRay(X, Y));
                image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
            }
        }
    }
private:
    Hit firstIntersect(Ray ray) {
        Hit bestHit;
        for (Intersectable * object : objects) {
            Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
        }
        if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
        return bestHit;
    }

    bool shadowIntersect(Ray ray) {	// for directional lights
        for (Intersectable * object : objects) if (object->intersect(ray).t > 0) return true;
        return false;
    }

    vec3 trace(Ray ray, int depth = 0) {
        Hit hit = firstIntersect(ray);
        if (hit.t < 0) return La;
        vec3 outRadiance = hit.material->ka * La;
        for (Light * light : lights) {
            Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
            float cosTheta = dot(hit.normal, light->direction);
            if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
                outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
                vec3 halfway = normalize(-ray.dir + light->direction);
                float cosDelta = dot(hit.normal, halfway);
                if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
            }
        }
        return outRadiance;
    }
};

class FullScreenTexturedQuad {
    unsigned int vao, vbo;
    Texture * texture;
    float vertexCoords[8] = { -1, -1, 1, -1, 1, 1, -1, 1 };
public:
    FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
    {
        setTexture(windowWidth, windowHeight, image);
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    }

    void setTexture(int windowWidth, int windowHeight, std::vector<vec4>& image) {
        texture = new Texture(windowWidth, windowHeight, image);
    }

    void Draw() {
        glBindVertexArray(vao);
        gpuProgram.setUniform(*texture, "textureUnit");
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }
};

Scene scene;
FullScreenTexturedQuad * fullScreenTexturedQuad;

void onInitialization() {
    vec3 kd(0.3f, 0.0f, 0.0f), ks(2, 2, 2);
    Material * material = new Material(kd, ks, 50);
    Triangle ta = Triangle(vec3(-5.0f, -5.0f, -5.0f), vec3(5.0f, -5.0f, -5.0f), vec3(0.0f, 5.0f, 0.0f), material);
    ta.intersect(Ray(eye, lookat-eye));

    glViewport(0, 0, windowWidth, windowHeight);
    camera.set(eye, lookat, vup, fov);
    scene.build();
    std::vector<vec4> image(windowWidth * windowHeight);
    scene.render(image);
    fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {
    fullScreenTexturedQuad->Draw();
    glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouse(int button, int state, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

long lastTime = 0;
long fok = -45;
void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME);
    if((time - lastTime) > 10) {
        lastTime = time;
        eye.x = cosf((fok + 90) / 180.0f * M_PI) * 2.0f;
        eye.z = sinf((fok + 90) / 180.0f * M_PI) * 2.0f;
        camera.set(eye, lookat, vup, fov);
        std::vector<vec4> image(windowWidth * windowHeight);
        scene.render(image);
        fullScreenTexturedQuad->setTexture(windowWidth, windowHeight, image);
        glutPostRedisplay();
        fok += 10;
    }
}