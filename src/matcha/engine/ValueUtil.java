package matcha.engine;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class ValueUtil {
/*
	public static Value div(final Value thiz, final Value x) {
		return new ValueMul(thiz, new ValuePow(x, -1));
	}

	public static Value div(final Value thiz, final double x) {
		return div(thiz, new ValueDouble(x));
	}
*/
/*
	public static Value sub(final Value thiz, final Value x) {
		return new ValueAdd(thiz, new ValueMul(x, new ValueDouble(-1)));
	}

	public static Value sub(final Value thiz, final double x) {
		return sub(thiz, new ValueDouble(x));
	}
*/
	/**
	 * Performs backpropagation on this value, computing the gradient of all linked previous values.
	 */
	public static void backward(final Value thiz)
	{
		final List<Value> ordering = new ArrayList<>();
		buildTopo(thiz, new HashSet<>(), ordering);
		Collections.reverse(ordering);
		
		thiz.grad = 1.0;
		for( final Value val : ordering ) {
			val.backward_pass();
		}
	}

	/**
	 * Build a topological-sorted ordering of children Values starting from this Value
	 */
	static void buildTopo(final Value parent, final Set<Value> visited, final List<Value> ordering)
	{
		if( !visited.contains(parent) ) {
			visited.add(parent);
			if( parent.thiz != null ) {
				buildTopo(parent.thiz, visited, ordering);
				if( parent.x != null ) {
					buildTopo(parent.x, visited, ordering);
				}
			}
			ordering.add(parent);
		}
	}

}
