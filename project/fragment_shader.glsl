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
    vec3 texColor = texture(texture1, TexCoord).rgb;
    vec3 color = pow(texColor, vec3(gamma)); // linéarisation

    vec3 ambient = 0.1 * color;
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * color;

    vec3 result = ambient + diffuse;
    result = pow(result, vec3(1.0/gamma)); // correction gamma

    FragColor = vec4(result, 1.0);
}
