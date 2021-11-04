#version 450 			// use glsl 4.5

// values coming in for each individual vertex
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 col;
layout(location = 2) in vec2 tex;

// uniform across all vertices
layout(set = 0, binding = 0) uniform UboViewProjection {
	mat4 projection;
	mat4 view;
} uboViewProjection;

// NOT IN USE, LEFT FOR REFERECE (using pushModel instead)
// Dynamic descriptor needs its own binding
layout(set = 0, binding = 1) uniform UboModel {
	mat4 model;
} uboModel;

// can only have one push constant block
layout(push_constant) uniform PushModel {
	mat4 model;
} pushModel;

// values going out for each individual vertex
layout(location = 0) out vec3 fragCol;
layout(location = 1) out vec2 fragTex;

void main() {
	// View transformations
	gl_Position = uboViewProjection.projection * uboViewProjection.view * pushModel.model * vec4(pos, 1.0);
	
	fragCol = col;
	fragTex = tex;
}