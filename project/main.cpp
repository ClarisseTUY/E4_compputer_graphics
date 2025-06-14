#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>

// ========== UTILS MATRICES 4x4 (col-major) ==========
void loadIdentity(float* mat) {
    for (int i=0; i<16; ++i) mat[i] = 0.0f;
    mat[0]=mat[5]=mat[10]=mat[15]=1.0f;
}

void multiplyMatrix(const float* A, const float* B, float* result) {
    for (int col=0; col<4; ++col) {
        for (int row=0; row<4; ++row) {
            result[col*4+row] = 0.0f;
            for (int i=0; i<4; ++i) {
                result[col*4+row] += A[i*4+row]*B[col*4+i];
            }
        }
    }
}

void translationMatrix(float x, float y, float z, float* mat) {
    loadIdentity(mat);
    mat[12] = x; mat[13] = y; mat[14] = z;
}

void rotationMatrixX(float angleDeg, float* mat) {
    float rad = angleDeg * (3.14159265f/180.0f);
    float c = cosf(rad), s = sinf(rad);
    loadIdentity(mat);
    mat[5] = c; mat[6] = s;
    mat[9] = -s; mat[10] = c;
}

void rotationMatrixY(float angleDeg, float* mat) {
    float rad = angleDeg * (3.14159265f/180.0f);
    float c = cosf(rad), s = sinf(rad);
    loadIdentity(mat);
    mat[0] = c; mat[2] = -s;
    mat[8] = s; mat[10] = c;
}

// Normalisation d’un vecteur 3D
void normalize(float* v) {
    float len = sqrtf(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    if(len > 0.00001f) {
        v[0]/=len; v[1]/=len; v[2]/=len;
    }
}

// Produit vectoriel v1 x v2 -> result
void cross(const float* v1, const float* v2, float* result) {
    result[0] = v1[1]*v2[2]-v1[2]*v2[1];
    result[1] = v1[2]*v2[0]-v1[0]*v2[2];
    result[2] = v1[0]*v2[1]-v1[1]*v2[0];
}

// Produit scalaire v1 . v2
float dot(const float* v1, const float* v2) {
    return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2];
}

// Construction de la View Matrix “LookAt”
void lookAt(const float* eye, const float* target, const float* up, float* view) {
    float f[3] = {target[0]-eye[0], target[1]-eye[1], target[2]-eye[2]};
    normalize(f);
    // forward = -f
    float forward[3] = {-f[0], -f[1], -f[2]};
    float right[3];
    cross(up, forward, right);
    normalize(right);
    float upCorrected[3];
    cross(forward, right, upCorrected);

    loadIdentity(view);
    // Colonnes = x,y,z axes du repère caméra transposés
    view[0] = right[0];    view[4] = right[1];    view[8] = right[2];
    view[1] = upCorrected[0]; view[5] = upCorrected[1]; view[9] = upCorrected[2];
    view[2] = forward[0];  view[6] = forward[1];  view[10] = forward[2];

    // Translation inverse
    view[12] = -dot(right, eye);
    view[13] = -dot(upCorrected, eye);
    view[14] = -dot(forward, eye);
}

// Projection perspective (fov en degrés, ratio = width/height, near, far)
void perspective(float fov, float ratio, float near, float far, float* proj) {
    float f = 1.0f / tanf((fov*3.14159265f/180.0f)/2.0f);
    loadIdentity(proj);
    proj[0] = f/ratio;
    proj[5] = f;
    proj[10] = (far+near)/(near-far);
    proj[11] = -1.0f;
    proj[14] = (2*far*near)/(near-far);
    proj[15] = 0.0f;
}

// ========== GESTION SOURIS pour rotation cube ==========
float rotX = 0.0f, rotY = 0.0f;
double lastX = 0.0, lastY = 0.0;
bool firstMouse = true;

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }
    double xoffset = xpos - lastX;
    double yoffset = ypos - lastY;
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.3f;
    rotX += (float)yoffset * sensitivity;
    rotY += (float)xoffset * sensitivity;

    if(rotX > 89.0f) rotX = 89.0f;
    if(rotX < -89.0f) rotX = -89.0f;
}

// ========== GESTION CLAVIER pour rotation cube avec ZQSD ==========


void processInput(GLFWwindow* window) {
    float sensitivity = 0.05f; // vitesse rotation

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)  // touche "Z" en AZERTY
        rotX -= sensitivity;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        rotX += sensitivity;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)  // touche "Q" en AZERTY
        rotY -= sensitivity;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        rotY += sensitivity;


    // Limite pour rotX
    if(rotX > 89.0f) rotX = 89.0f;
    if(rotX < -89.0f) rotX = -89.0f;
}

// ========== SHADERS ==========
const char* vertexShaderSource = R"glsl(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aTexCoord;

uniform mat4 WorldMatrix;
uniform mat4 ViewMatrix;
uniform mat4 ProjectionMatrix;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

void main() {
    mat4 WV = ViewMatrix * WorldMatrix;
    FragPos = vec3(WV * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(WV))) * aNormal;
    TexCoord = aTexCoord;
    gl_Position = ProjectionMatrix * WV * vec4(aPos, 1.0);
}
)glsl";

const char* fragmentShaderSource = R"glsl(
#version 330 core
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D texture1;
uniform vec3 lightPos;
uniform vec3 viewPos;

const float gamma = 2.2;

void main() {
    // Gamma correction - lire la texture linéairement
    vec3 texColor = texture(texture1, TexCoord).rgb;
    // linéarisation (on suppose la texture est sRGB, on corrige en puissance gamma)
    vec3 color = pow(texColor, vec3(gamma));

    // Lumière simple diffuse + ambient
    vec3 ambient = 0.1 * color;

    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * color;

    vec3 result = ambient + diffuse;

    // Correction gamma à la sortie (compression)
    result = pow(result, vec3(1.0/gamma));

    FragColor = vec4(result, 1.0);
}
)glsl";

// ========== FONCTION AIDE COMPIL SHADERS ==========
GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint success; glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if(!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        std::cerr << "Shader compilation error:\n" << log << std::endl;
    }
    return shader;
}

GLuint createProgram(const char* vertSrc, const char* fragSrc) {
    GLuint vert = compileShader(GL_VERTEX_SHADER, vertSrc);
    GLuint frag = compileShader(GL_FRAGMENT_SHADER, fragSrc);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vert);
    glAttachShader(prog, frag);
    glLinkProgram(prog);

    GLint success; glGetProgramiv(prog, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(prog, 512, nullptr, log);
        std::cerr << "Program link error:\n" << log << std::endl;
    }

    glDeleteShader(vert);
    glDeleteShader(frag);

    return prog;
}

// ========== MAIN ==========
int main() {
    if(!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(800,600,"Cube texturé avec lumière & gamma correction", nullptr, nullptr);
    if(!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if(glewInit() != GLEW_OK) {
        std::cerr << "Failed to init GLEW\n";
        return -1;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_FRAMEBUFFER_SRGB);

    // Plus besoin du callback souris
    glfwSetCursorPosCallback(window, mouse_callback);

    GLuint shaderProgram = createProgram(vertexShaderSource, fragmentShaderSource);


    // Cube data (positions, normals, texcoords)
    float vertices[] = {
        // positions          // normals           // texcoords
        -0.5f,-0.5f,-0.5f,   0,0,-1,             0,0,
         0.5f,-0.5f,-0.5f,   0,0,-1,             1,0,
         0.5f, 0.5f,-0.5f,   0,0,-1,             1,1,
         0.5f, 0.5f,-0.5f,   0,0,-1,             1,1,
        -0.5f, 0.5f,-0.5f,   0,0,-1,             0,1,
        -0.5f,-0.5f,-0.5f,   0,0,-1,             0,0,

        -0.5f,-0.5f, 0.5f,   0,0,1,              0,0,
         0.5f,-0.5f, 0.5f,   0,0,1,              1,0,
         0.5f, 0.5f, 0.5f,   0,0,1,              1,1,
         0.5f, 0.5f, 0.5f,   0,0,1,              1,1,
        -0.5f, 0.5f, 0.5f,   0,0,1,              0,1,
        -0.5f,-0.5f, 0.5f,   0,0,1,              0,0,

        -0.5f, 0.5f, 0.5f,  -1,0,0,              1,0,
        -0.5f, 0.5f,-0.5f,  -1,0,0,              1,1,
        -0.5f,-0.5f,-0.5f,  -1,0,0,              0,1,
        -0.5f,-0.5f,-0.5f,  -1,0,0,              0,1,
        -0.5f,-0.5f, 0.5f,  -1,0,0,              0,0,
        -0.5f, 0.5f, 0.5f,  -1,0,0,              1,0,

         0.5f, 0.5f, 0.5f,   1,0,0,              1,0,
         0.5f, 0.5f,-0.5f,   1,0,0,              1,1,
         0.5f,-0.5f,-0.5f,   1,0,0,              0,1,
         0.5f,-0.5f,-0.5f,   1,0,0,              0,1,
         0.5f,-0.5f, 0.5f,   1,0,0,              0,0,
         0.5f, 0.5f, 0.5f,   1,0,0,              1,0,

        -0.5f,-0.5f,-0.5f,   0,-1,0,             0,1,
         0.5f,-0.5f,-0.5f,   0,-1,0,             1,1,
         0.5f,-0.5f, 0.5f,   0,-1,0,             1,0,
         0.5f,-0.5f, 0.5f,   0,-1,0,             1,0,
        -0.5f,-0.5f, 0.5f,   0,-1,0,             0,0,
        -0.5f,-0.5f,-0.5f,   0,-1,0,             0,1,

        -0.5f, 0.5f,-0.5f,   0,1,0,              0,1,
         0.5f, 0.5f,-0.5f,   0,1,0,              1,1,
         0.5f, 0.5f, 0.5f,   0,1,0,              1,0,
         0.5f, 0.5f, 0.5f,   0,1,0,              1,0,
        -0.5f, 0.5f, 0.5f,   0,1,0,              0,0,
        -0.5f, 0.5f,-0.5f,   0,1,0,              0,1
    };

    GLuint VBO, VAO;
    glGenVertexArrays(1,&VAO);
    glGenBuffers(1,&VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // positions
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)0);
    glEnableVertexAttribArray(0);
    // normals
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    // texcoords
    glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)(6*sizeof(float)));
    glEnableVertexAttribArray(2);

    // Chargement texture (simple couleur générée ici)
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    unsigned char texData[3] = {255, 128, 0}; // orange
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1,1,0, GL_RGB, GL_UNSIGNED_BYTE, texData);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glUseProgram(shaderProgram);
    glUniform1i(glGetUniformLocation(shaderProgram, "texture1"), 0);

    int locWorld = glGetUniformLocation(shaderProgram, "WorldMatrix");
    int locView = glGetUniformLocation(shaderProgram, "ViewMatrix");
    int locProj = glGetUniformLocation(shaderProgram, "ProjectionMatrix");
    int locLight = glGetUniformLocation(shaderProgram, "lightPos");
    int locViewPos = glGetUniformLocation(shaderProgram, "viewPos");

    
    float projection[16];
    perspective(45.0f, 800.0f/600.0f, 0.1f, 100.0f, projection);

    float view[16];
    float eye[3] = {0.0f, 0.0f, 3.0f};
    float target[3] = {0.0f, 0.0f, 0.0f};
    float up[3] = {0.0f, 1.0f, 0.0f};

    while(!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Gérer les entrées clavier ZQSD pour rotation
        processInput(window);

        lookAt(eye, target, up, view);

        // Rotation cube selon les rotations rotX, rotY modifiées au clavier
        float Rx[16], Ry[16], WorldRot[16];
        rotationMatrixX(rotX, Rx);
        rotationMatrixY(rotY, Ry);
        multiplyMatrix(Ry, Rx, WorldRot);

        float trans[16];
        translationMatrix(0.0f, 0.0f, 0.0f, trans);

        float World[16];
        multiplyMatrix(trans, WorldRot, World);

        glUseProgram(shaderProgram);
        glUniformMatrix4fv(locWorld, 1, GL_FALSE, World);
        glUniformMatrix4fv(locView, 1, GL_FALSE, view);
        glUniformMatrix4fv(locProj, 1, GL_FALSE, projection);

        glUniform3f(locLight, 1.0f, 1.0f, 2.0f);
        glUniform3f(locViewPos, eye[0], eye[1], eye[2]);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);

        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup (inchangé)
    glDeleteVertexArrays(1,&VAO);
    glDeleteBuffers(1,&VBO);
    glDeleteProgram(shaderProgram);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}