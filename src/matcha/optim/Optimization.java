package matcha.optim;

import java.util.List;
import matcha.engine.Value;

public abstract class Optimization {

	private final List<Value> params;

	public Optimization(final List<Value> params) {
		this.params = params;
	}
	
	public void zeroGrad() {
		params.forEach( param -> param.setGradient(0.0));
	}
}
