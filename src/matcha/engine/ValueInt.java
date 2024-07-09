package matcha.engine;

public class ValueInt extends Value {

	public ValueInt(final int data) {
		this.data = data;
	}

	@Override
	public void backward_pass() {
	}
}

