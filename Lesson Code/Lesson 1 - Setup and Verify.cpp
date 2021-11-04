#define GLFW_INCLUDE_VULKAN /*This causes to GLFW to automatically include vulkan.*/
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS /*Use radians*/
#define GLM_FORCE_DEPTH_ZERO_TO_ONE /*Use 0 to 1 instead of -1  to 1*/
#include <glm.hpp>
#include <mat4x4.hpp>

int main() {
	glfwInit();

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); /*Tells GLFW not to use opengl or other API*/
	GLFWwindow* window = glfwCreateWindow(800, 600, "Test Window", nullptr, nullptr);

	/*How many extensions can this vulkan instance can support.*/
	uint32_t extensionCount = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

	/*Print number of extensions supported by this instance*/
	printf("Extension count: %i\n", extensionCount);

	/*Test to make sure there are no errors with glm*/
	glm::mat4 testMatrix(1.0f);
	glm::vec4 testVector(1.0f);
	auto testResult = testMatrix * testVector;

	while (!glfwWindowShouldClose(window)) {

		glfwPollEvents(); /*Check if x on windows has been clicked*/
	}

	glfwDestroyWindow(window); /*clean up window*/

}