package matcha.nn;

public enum Activation {

	none,
	relu,
	tanh;

	public static Activation activation(final Activation activation) {
		if (activation == null) {
			throw new RuntimeException(
				String.format("Warning: activation function must be of the following: '%s', '%s', '%s'",
				none, relu, tanh));
		}
		return activation;
	}
}
