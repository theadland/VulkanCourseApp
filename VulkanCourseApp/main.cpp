#define STB_IMAGE_IMPLEMENTATION	// activates stb image
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // glm defaults to -1 to 1 because of openGl

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <stdexcept>
#include <vector>
#include <iostream>

#include "VulkanRenderer.h"

GLFWwindow* window;
VulkanRenderer VulkanRenderer;

/*This function will create the window. In the future, probably a good idea to make a window class.*/
void initWindow(std::string wName = "Test Window", const int width = 800, const int height = 600) {
	// Initialize GLFW
	glfwInit();

	// Set GLFW to Not work with OpenGL
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // simplify things for now

	window = glfwCreateWindow(width, height, wName.c_str(), nullptr, nullptr);

}

int main() {
	
	// Create Window
	initWindow("Test Window", 800, 600);

	// Create Vulkan Renderer Instance, if init fails return failure
	if (VulkanRenderer.init(window) == EXIT_FAILURE)
	{ 
		return EXIT_FAILURE;
	}

	float angle = 0.0f;
	float deltaTime = 0.0f;
	float lastTime = 0.0f;

	// Loop until close
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) == GLFW_TRUE)
			continue;




		float now = glfwGetTime();
		deltaTime = now - lastTime;
		lastTime = now;

		angle += 10.0f * deltaTime;
		if (angle > 360.0f) { angle -= 360.0f; }

		glm::mat4 firstModel(1.0f);
		glm::mat4 secondModel(1.0f);

		firstModel = glm::translate(firstModel, glm::vec3(0.0f, 0.0f, -3.0f));
		firstModel = glm::rotate(firstModel, glm::radians(angle), glm::vec3(0.0f, 0.0f, 1.0f));

		secondModel = glm::translate(secondModel, glm::vec3(0.0f, 0.0f, -4.0f));
		secondModel = glm::rotate(secondModel, glm::radians(-angle * 10), glm::vec3(0.0f, 0.0f, 1.0f));

		VulkanRenderer.updateModel(0, firstModel);
		VulkanRenderer.updateModel(1, secondModel);



		VulkanRenderer.draw();
	}

	VulkanRenderer.cleanup();

	glfwDestroyWindow(window);
	glfwTerminate();


}