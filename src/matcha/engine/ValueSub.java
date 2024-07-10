package matcha.engine;

//Tom Sawyer solution, it seems to work

public class ValueSub extends Value {

	public ValueSub(final Value thiz, final Value x) {
		this.data = thiz.data - x.data;
		this.thiz = thiz;
		this.x = x;
	}

	@Override
	public void backward_pass() {
		thiz.grad += 1.0 * grad;
		x.grad += 1.0 * grad;
	}
}

