from utitlities import calculate_lip_constant, compute_gradient, scaled_denoiser, compute_function_error
import numpy as np


class DcNN_Fista:
    """
    Fast iterative shrinking/thresholding algorithm with the help of
    an image denoiser as an imaging prior
    """

    def __init__(self, iterations, scaler, path_to_model, lambda_):
        self.iterations = iterations
        self.scaler = scaler
        self.path_to_model = path_to_model
        self.lambda_ = lambda_

    def dcnn_fista(self, A, b, original_img, lip_constant):

        (n_samples, n_features) = A.shape
        n_kernels = int(n_features / n_samples)
        self.n_samples, self.n_kernels = n_samples, n_kernels

        if lip_constant is None:
            lip_constant = calculate_lip_constant(A)

        lipinv = 1 / lip_constant
        lambdaLipinv = self.lambda_ * lipinv

        # Initializing the coefficients (Step 0)
        x_current = np.zeros(n_features, dtype=np.float)
        x_next = np.zeros(n_features, dtype=np.float)
        y_current = np.copy(x_next)
        y_next = np.zeros(n_features, dtype=np.float)
        tau_old = 1.0
        tau_new = 0.0

        patience = 30
        previous_error = 10 ** 4

        # A loop for the algorithm
        for i in range(self.iterations):
            # Compute the gradient
            tmp_grad = compute_gradient(A, b, y_current)
            grad = y_current - lipinv * tmp_grad

            # Apply projection
            x_next = scaled_denoiser(grad, self.scaler, self.path_to_model)

            tau_new = (1.0 + np.sqrt(1.0 + 4.0 * tau_old ** 2)) / 2.0

            y_next = x_next + (tau_old - 1) / tau_new * (x_next - x_current)

            # Update
            x_current = x_next
            tau_old = tau_new
            y_current = y_next

            error = compute_function_error(original_img, x_next)

            # Break if there is no change in MSE
            if error < previous_error:
                print('Iteration ' + '{}'.format(i + 1) + ' '
                      + '{:.2f}'.format(error) + ' Yes')
                previous_error = error
            else:
                print('Iteration ' + '{}'.format(i + 1) + ' '
                      + '{:.2f}'.format(error) + ' No')
                if patience == 0:
                    break
                else:
                    patience -= 1

        return x_next, i + 1
