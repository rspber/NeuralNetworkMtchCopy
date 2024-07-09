package matcha.optim;

import matcha.engine.Value;
import java.util.List;

public class SGD extends Optimization {

	private final List<Value> params;
	private final double lr;

	public SGD(final List<Value> params, final double lr) {
		super(params);
		this.params = params;
		this.lr = lr;
	}

	public void step() {
		params.forEach( param -> param.step(-lr));
	}

}
