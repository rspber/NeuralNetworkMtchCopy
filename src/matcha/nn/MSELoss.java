package matcha.nn;

import matcha.engine.Value;
import matcha.engine.ValueAdd;
import matcha.engine.ValueDouble;
import matcha.engine.ValuePow;
import matcha.engine.ValueUtil;

public class MSELoss extends Loss<Value>{
	public MSELoss() {

	}

	public Value loss(Value[] outputs, Value[] targets) {
		Value loss = new ValueDouble(0.0);
		for(int i = 0; i < targets.length; i++) {
			loss = new ValueAdd(loss, new ValuePow((ValueUtil.sub(outputs[i], targets[i])), 2));
		}

		return loss;
	}
}
