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
