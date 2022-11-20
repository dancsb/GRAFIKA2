//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
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
// Nev    : Dancs Balázs
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

float fov = 45.0f / 180.0f * M_PI;
int angle = -205;

GPUProgram gpuProgram;

const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1);
	}
)";

const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;
	out vec4 fragmentColor;

	void main() {
		fragmentColor = texture(textureUnit, texcoord);
	}
)";

struct Ray {
    vec3 start, dir;
    Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

struct Hit {
    float t;
    vec3 position, normal;
    Hit() { t = -1; }
};

class Camera {
    vec3 eye, lookat, right, up;
public:
    Camera() {
        set();
    }

    void set() {
        eye = vec3(cosf((float)(((float)angle + 90.0f) / 180.0f * M_PI)) * 2.0f, 0, sinf((float)(((float)angle + 90.0f) / 180.0f * M_PI)) * 2.0f);
        lookat = vec3(0, 0, 0);
        vec3 w = eye - lookat;
        float focus = length(w);
        right = normalize(cross(vec3(0, 1, 0), w)) * focus * tanf(fov / 2);
        up = normalize(cross(w, right)) * focus * tanf(fov / 2);
    }

    Ray getRay(int X, int Y) {
        vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
        return {eye, dir};
    }
};

Camera camera;

struct cone_data {
    vec3 p, n;
    float h, alpha;
    int id;

    cone_data(const vec3 &p, const vec3 &n, float h, float alpha, int id) : p(p), n(n), h(h), alpha(alpha), id(id) {}
};

std::vector<cone_data> cones;

class Intersectable {
protected:
    int color = 0;
public:
    virtual Hit intersect(const Ray& ray) = 0;
};

struct Triangle : public Intersectable {
    vec3 r1, r2 ,r3;

    Triangle(const vec3& r1, const vec3& r2, const vec3& r3) : r1(r1), r2(r2), r3(r3) {}

    Hit intersect(const Ray& ray) override {
        Hit hit;
        hit.normal = cross((r2 - r1), (r3 - r1));
        hit.t = dot((r1 - ray.start), hit.normal) / dot(ray.dir, hit.normal);
        hit.position = ray.start + ray.dir * hit.t;
        if (dot(cross((r2 - r1), (hit.position - r1)), hit.normal) < 0.0f) hit.t = -1.0f;
        if (dot(cross((r3 - r2), (hit.position - r2)), hit.normal) < 0.0f) hit.t = -1.0f;
        if (dot(cross((r1 - r3), (hit.position - r3)), hit.normal) < 0.0f) hit.t = -1.0f;
        return hit;
    }
};

struct Cone : public Intersectable {
    vec3 p, n;
    float h, alpha;

    Cone(const vec3 &p, const vec3 &n, float h, float alpha, int color) : p(p), n(n), h(h), alpha(alpha) {
        this->color = color;
    }

    Hit intersect(const Ray& ray) override {
        Hit hit;
        vec3 dist = ray.start - p;
        float a = powf(dot(n, ray.dir), 2.0f) - powf(cosf(alpha), 2.0f);
        float b = ((dot(n, ray.dir) * dot(dist, n)) - dot(dist, ray.dir) * powf(cosf(alpha), 2.0f)) * 2.0f;
        float c = powf(dot(dist, n), 2.0f) - dot(dist, dist) * powf(cosf(alpha), 2.0f);
        float discr = b * b - 4.0f * a * c;
        if (discr < 0.0f) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;
        float t2 = (-b - sqrt_discr) / 2.0f / a;
        vec3 hitpos1 = ray.start + ray.dir * t1;
        vec3 hitpos2 = ray.start + ray.dir * t2;
        if (dot(hitpos1 - p, n) < 0.0f || dot(hitpos1 - p, n) > h) t1 = -1.0f;
        if (dot(hitpos2 - p, n) < 0.0f || dot(hitpos2 - p, n) > h) t2 = -1.0f;
        if (t1 >= 0.0f && t2 < 0.0f) hit.t = t1;
        else if (t1 < 0.0f && t2 >= 0.0f) hit.t = t2;
        else hit.t = (t1 < t2) ? t1 : t2;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal =normalize(2.0f * (dot(hit.position - p, n)) * n - 2.0f * (hit.position - p) * powf(cosf(alpha), 2.0f));
        return hit;
    }
};

class Scene {
    std::vector<Intersectable *> objects;
    vec3 La;
    void buildRoom() {
        objects.push_back(new Triangle(vec3(-0.5f, -0.5f, -0.5f), vec3(0.5f, -0.5f, -0.5f), vec3(-0.5f, -0.5f, 0.5f)));
        objects.push_back(new Triangle(vec3(-0.5f, -0.5f, 0.5f), vec3(0.5f, -0.5f, -0.5f), vec3(0.5f, -0.5f, 0.5f)));

        objects.push_back(new Triangle(vec3(-0.5f, 0.5f, -0.5f), vec3(0.5f, 0.5f, -0.5f), vec3(-0.5f, 0.5f, 0.5f)));
        objects.push_back(new Triangle(vec3(-0.5f, 0.5f, 0.5f), vec3(0.5f, 0.5f, -0.5f), vec3(0.5f, 0.5f, 0.5f)));

        objects.push_back(new Triangle(vec3(-0.5f, 0.5f, -0.5f), vec3(-0.5f, -0.5f, -0.5f), vec3(-0.5f, 0.5f, 0.5f)));
        objects.push_back(new Triangle(vec3(-0.5f, 0.5f, 0.5f), vec3(-0.5f, -0.5f, -0.5f), vec3(-0.5f, -0.5f, 0.5f)));

        objects.push_back(new Triangle(vec3(-0.5f, 0.5f, -0.5f), vec3(-0.5f, -0.5f, -0.5f), vec3(0.5f, 0.5f, -0.5f)));
        objects.push_back(new Triangle(vec3(0.5f, 0.5f, -0.5f), vec3(-0.5f, -0.5f, -0.5f), vec3(0.5f, -0.5f, -0.5f)));

        objects.push_back(new Triangle(vec3(0.5f, 0.5f, -0.5f), vec3(0.5f, -0.5f, -0.5f), vec3(0.5f, 0.5f, 0.5f)));
        objects.push_back(new Triangle(vec3(0.5f, 0.5f, 0.5f), vec3(0.5f, -0.5f, -0.5f), vec3(0.5f, -0.5f, 0.5f)));

        objects.push_back(new Triangle(vec3(0.5f, 0.5f, 0.5f), vec3(0.5f, -0.5f, 0.5f), vec3(-0.5f, 0.5f, 0.5f)));
        objects.push_back(new Triangle(vec3(-0.5f, 0.5f, 0.5f), vec3(0.5f, -0.5f, 0.5f), vec3(-0.5f, -0.5f, 0.5f)));
    }
    void buildIcosahedron() {
        float scale = 0.2f;
        float a = 0.525731f * scale;
        float b = 0.850651f * scale;
        vec3 offset(0.25f, -0.5f + b, -0.25f);

        std::vector<vec3> vertices;

        vertices.push_back(vec3(0, -a, b) + offset);
        vertices.push_back(vec3(b, 0, a) + offset);
        vertices.push_back(vec3(b, 0, -a) + offset);
        vertices.push_back(vec3(-b, 0, -a) + offset);
        vertices.push_back(vec3(-b, 0, a) + offset);
        vertices.push_back(vec3(-a, b, 0) + offset);
        vertices.push_back(vec3(a, b, 0) + offset);
        vertices.push_back(vec3(a, -b, 0) + offset);
        vertices.push_back(vec3(-a, -b, 0) + offset);
        vertices.push_back(vec3(0, -a, -b) + offset);
        vertices.push_back(vec3(0, a, -b) + offset);
        vertices.push_back(vec3(0, a, b) + offset);

        objects.push_back(new Triangle(vertices[1], vertices[2], vertices[6]));
        objects.push_back(new Triangle(vertices[1], vertices[7], vertices[2]));
        objects.push_back(new Triangle(vertices[3], vertices[4], vertices[5]));
        objects.push_back(new Triangle(vertices[4], vertices[3], vertices[8]));
        objects.push_back(new Triangle(vertices[6], vertices[5], vertices[11]));
        objects.push_back(new Triangle(vertices[5], vertices[6], vertices[10]));
        objects.push_back(new Triangle(vertices[9], vertices[10], vertices[2]));
        objects.push_back(new Triangle(vertices[10], vertices[9], vertices[3]));
        objects.push_back(new Triangle(vertices[7], vertices[8], vertices[9]));
        objects.push_back(new Triangle(vertices[8], vertices[7], vertices[0]));
        objects.push_back(new Triangle(vertices[11], vertices[0], vertices[1]));
        objects.push_back(new Triangle(vertices[0], vertices[11], vertices[4]));
        objects.push_back(new Triangle(vertices[6], vertices[2], vertices[10]));
        objects.push_back(new Triangle(vertices[1], vertices[6], vertices[11]));
        objects.push_back(new Triangle(vertices[3], vertices[5], vertices[10]));
        objects.push_back(new Triangle(vertices[5], vertices[4], vertices[11]));
        objects.push_back(new Triangle(vertices[2], vertices[7], vertices[9]));
        objects.push_back(new Triangle(vertices[7], vertices[1], vertices[0]));
        objects.push_back(new Triangle(vertices[3], vertices[9], vertices[8]));
        objects.push_back(new Triangle(vertices[4], vertices[8], vertices[0]));
    }
    void buildDodecahedron() {
        float scale = 0.25f;
        float a = 0.356822f * scale;
        float b = 0.57735f * scale;
        float c = 0.934172f * scale;
        vec3 offset(-0.15f, -0.5f + c, 0.15f);

        std::vector<vec3> vertices;

        vertices.push_back(vec3(-b, -b, b) + offset);
        vertices.push_back(vec3(c, a, 0) + offset);
        vertices.push_back(vec3(c, -a, 0) + offset);
        vertices.push_back(vec3(-c, a, 0) + offset);
        vertices.push_back(vec3(-c, -a, 0) + offset);
        vertices.push_back(vec3(0, c, a) + offset);
        vertices.push_back(vec3(0, c, -a) + offset);
        vertices.push_back(vec3(a, 0, -c) + offset);
        vertices.push_back(vec3(-a, 0, -c) + offset);
        vertices.push_back(vec3(0, -c, -a) + offset);
        vertices.push_back(vec3(0, -c, a) + offset);
        vertices.push_back(vec3(a, 0, c) + offset);
        vertices.push_back(vec3(-a, 0, c) + offset);
        vertices.push_back(vec3(b, b, -b) + offset);
        vertices.push_back(vec3(b, b, b) + offset);
        vertices.push_back(vec3(-b, b, -b) + offset);
        vertices.push_back(vec3(-b, b, b) + offset);
        vertices.push_back(vec3(b, -b, -b) + offset);
        vertices.push_back(vec3(b, -b, b) + offset);
        vertices.push_back(vec3(-b, -b, -b) + offset);

        objects.push_back(new Triangle(vertices[18], vertices[2], vertices[1]));
        objects.push_back(new Triangle(vertices[11], vertices[18], vertices[1]));
        objects.push_back(new Triangle(vertices[14], vertices[11], vertices[1]));
        objects.push_back(new Triangle(vertices[7], vertices[13], vertices[1]));
        objects.push_back(new Triangle(vertices[17], vertices[7], vertices[1]));
        objects.push_back(new Triangle(vertices[2], vertices[17], vertices[1]));
        objects.push_back(new Triangle(vertices[19], vertices[4], vertices[3]));
        objects.push_back(new Triangle(vertices[8], vertices[19], vertices[3]));
        objects.push_back(new Triangle(vertices[15], vertices[8], vertices[3]));
        objects.push_back(new Triangle(vertices[12], vertices[16], vertices[3]));
        objects.push_back(new Triangle(vertices[0], vertices[12], vertices[3]));
        objects.push_back(new Triangle(vertices[4], vertices[0], vertices[3]));
        objects.push_back(new Triangle(vertices[6], vertices[15], vertices[3]));
        objects.push_back(new Triangle(vertices[5], vertices[6], vertices[3]));
        objects.push_back(new Triangle(vertices[16], vertices[5], vertices[3]));
        objects.push_back(new Triangle(vertices[5], vertices[14], vertices[1]));
        objects.push_back(new Triangle(vertices[6], vertices[5], vertices[1]));
        objects.push_back(new Triangle(vertices[13], vertices[6], vertices[1]));
        objects.push_back(new Triangle(vertices[9], vertices[17], vertices[2]));
        objects.push_back(new Triangle(vertices[10], vertices[9], vertices[2]));
        objects.push_back(new Triangle(vertices[18], vertices[10], vertices[2]));
        objects.push_back(new Triangle(vertices[10], vertices[0], vertices[4]));
        objects.push_back(new Triangle(vertices[9], vertices[10], vertices[4]));
        objects.push_back(new Triangle(vertices[19], vertices[9], vertices[4]));
        objects.push_back(new Triangle(vertices[19], vertices[8], vertices[7]));
        objects.push_back(new Triangle(vertices[9], vertices[19], vertices[7]));
        objects.push_back(new Triangle(vertices[17], vertices[9], vertices[7]));
        objects.push_back(new Triangle(vertices[8], vertices[15], vertices[6]));
        objects.push_back(new Triangle(vertices[7], vertices[8], vertices[6]));
        objects.push_back(new Triangle(vertices[13], vertices[7], vertices[6]));
        objects.push_back(new Triangle(vertices[11], vertices[14], vertices[5]));
        objects.push_back(new Triangle(vertices[12], vertices[11], vertices[5]));
        objects.push_back(new Triangle(vertices[16], vertices[12], vertices[5]));
        objects.push_back(new Triangle(vertices[12], vertices[0], vertices[10]));
        objects.push_back(new Triangle(vertices[11], vertices[12], vertices[10]));
        objects.push_back(new Triangle(vertices[18], vertices[11], vertices[10]));
    }
public:
    void build() {
        La = vec3(0.0f, 0.0f, 0.0f);

        buildRoom();
        buildIcosahedron();
        buildDodecahedron();

        objects.push_back(new Cone(cones[0].p, cones[0].n, cones[0].h, cones[0].alpha, cones[0].id));
        objects.push_back(new Cone(cones[1].p, cones[1].n, cones[1].h, cones[1].alpha, cones[1].id));
        objects.push_back(new Cone(cones[2].p, cones[2].n, cones[2].h, cones[2].alpha, cones[2].id));
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
        Hit bestHit, secBestHit;
        for (Intersectable * object : objects) {
            Hit hit = object->intersect(ray);
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) {
                secBestHit = bestHit;
                bestHit = hit;
            }
            if(hit.t > bestHit.t && (secBestHit.t < 0 || hit.t < secBestHit.t))
                secBestHit = hit;
        }
        if (secBestHit.t > 0) bestHit = secBestHit;
        if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
        return bestHit;
    }

    Hit firstIntersectP(Ray ray) {
        Hit bestHit;
        for (Intersectable * object : objects) {
            Hit hit = object->intersect(ray);
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) {
                bestHit = hit;
            }
        }
        if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
        return bestHit;
    }

    bool shadowIntersect(Ray ray, int id) {
        Hit hit = firstIntersectP(ray);
        for (int i = 0; i < 3; i++)
            if (id == i + 1 && hit.t > 0 && length(hit.position - cones[i].p) > 0.002f) return true;
        return false;
    }

    vec3 trace(Ray ray) {
        Hit hit = firstIntersect(ray);
        if (hit.t < 0) return La;
        float c = 0.2f * (1.0f + dot(normalize(hit.normal), normalize(-ray.dir)));
        vec3 outRadiance = vec3(c, c, c);
        float rgb[3] = { 0 };

        for (int i = 0; i < 3; i++)
            if (!shadowIntersect(Ray(hit.position + hit.normal * 0.0002f, normalize(cones[i].p + cones[i].n * 0.002f - hit.position)), cones[i].id))
                rgb[i] = c / length(hit.position - cones[i].p);

        outRadiance = outRadiance + vec3(rgb[0], rgb[1], rgb[2]);
        return outRadiance;
    }
};

class FullScreenTexturedQuad {
    unsigned int vao, vbo;
    Texture * texture;
public:
    FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
    {
        setTexture(windowWidth, windowHeight, image);
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float vertexCoords[] = { -1, -1, 1, -1, 1, 1, -1, 1 };
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
    cones.push_back(cone_data(vec3(0.0f, 0.5f, 0.0f), vec3(0, -1, 0), 0.2f, 20.0f / 180.0f * M_PI, 1));
    cones.push_back(cone_data(vec3(0.3f, 0.1f, -0.5f), vec3(0, 0, 1), 0.2f, 20.0f / 180.0f * M_PI, 2));
    cones.push_back(cone_data(vec3(-0.5f, 0.0f, 0.0f), vec3(1, 0, 0), 0.2f, 20.0f / 180.0f * M_PI, 3));

    glViewport(0, 0, windowWidth, windowHeight);
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

void onIdle() {
    angle -= 5;
    if (angle < -360) angle += 360;
    camera.set();
    std::vector<vec4> image(windowWidth * windowHeight);
    scene.render(image);
    fullScreenTexturedQuad->setTexture(windowWidth, windowHeight, image);
    glutPostRedisplay();
}
