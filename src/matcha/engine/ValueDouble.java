package matcha.engine;

public class ValueDouble extends Value {

	public ValueDouble(final double data) {
		this.data = data;
	}

	@Override
	public void backward_pass() {
	}
}

