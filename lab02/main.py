import cv2
import numpy


EPSILON = 1e-12


def convolution(data: numpy.ndarray, kernels: numpy.ndarray) -> numpy.ndarray:
    data_channels, data_height, data_width = data.shape
    kernels_count, kernel_channels, kernel_height, kernel_width = kernels.shape
    result = numpy.zeros(shape=(kernels_count, data_height, data_width))

    assert (data_channels == kernel_channels)

    for kernel in range(kernels_count):
        for y in range(data_height):
            for x in range(data_width):
                for i in range(-kernel_height // 2, kernel_height // 2):
                    for j in range(-kernel_width // 2, kernel_width // 2):
                        for k in range(data_channels):
                            result[kernel, y, x] += \
                                data[k, max(y + i, 0), min(x + i, data_width - 1)] * kernels[kernel, k, i, j]
    return result


def normalize(data, beta, gamma):
    result = numpy.zeros(shape=data.shape)
    data_channels, data_height, data_width = data.shape

    mu = numpy.average(data, axis=(1, 2))
    standard_deviation = numpy.std(data, axis=(1, 2))
    for channel in range(data_channels):
        result[channel, :, :] = (data[channel, :, :] - mu[channel]) / (standard_deviation[channel] + EPSILON)
        result[channel, :, :] = gamma[channel, :, :] * result[channel, :, :] + beta[channel, :, :]
    return result


def relu(data):
    return numpy.maximum(data, 0)


def max_pooling(data):
    data_channels, data_height, data_width = data.shape

    kernel_height = 2
    kernel_width = 2

    result_height = data_height // kernel_height
    result_width = data_width // kernel_width

    result = numpy.zeros(shape=(data_channels, result_height, result_width))

    for channel in range(data_channels):
        for y in range(result_height):
            for x in range(result_width):
                y_from = y * kernel_height
                y_to = y * kernel_height + kernel_height
                x_from = x * kernel_width
                x_to = x * kernel_width + kernel_width
                result[channel, y, x] = numpy.amax(data[channel, y_from:y_to, x_from:x_to], axis=(0, 1))
    return result


def softmax(data):
    temp_data = numpy.reshape(data, -1)

    data_exponential = numpy.exp(temp_data)
    data_exponential_sum = numpy.sum(data_exponential)
    return data_exponential / data_exponential_sum


def main():
    image = cv2.imread('./Cat.png')
    data = numpy.transpose(image, (2, 1, 0))
    data = convolution(data, numpy.random.rand(5, 3, 3, 3))
    data = normalize(data, numpy.random.uniform(2, 8, data.shape), numpy.random.uniform(2, 8, data.shape))
    data = max_pooling(data)
    prediction = softmax(data)


if __name__ == '__main__':
    main()
