package matcha.test;

import java.util.ArrayList;
import java.util.List;

import matcha.engine.Value;
import matcha.engine.ValueUtil;
import matcha.nn.Linear;
import matcha.nn.Module;
import matcha.nn.MSELoss;
import matcha.nn.ReLU;
import matcha.nn.Sequential;
import matcha.nn.Tanh;
import matcha.optim.SGD;

public class Example{

	public static void main(String[] args) {
		// constructing a simple network
		final List<Module<Value[]>> layers = new ArrayList<>();
		layers.add(new Linear(3,4));
		layers.add(new ReLU());
		layers.add(new Linear(4,4));
		layers.add(new Tanh());
		layers.add(new Linear(4,1));
		layers.add(new Tanh());

		final Sequential nn = new Sequential(layers);

		// prints network information, such as layers and dimensions
		System.out.println(nn);

		// example input data
		final double[][] Xs = new double[][] {
			{2.0, 3.0, -1.0},
			{3.0, -1.0, 0.5},
			{0.5, 1.0, 1.0},
			{1.0, 1.0, -1.0}
		};

		// example target values for each input
		double[] Ys = new double[]{1.0, -1.0, -1.0, 1.0};

		// check newly trained outputs
		final ArrayList<Value> outs = new ArrayList<>();
		for( int i=0; i < Xs.length; ++i ) {
			outs.add( nn.forward(Xs[i])[0] );
		}

		System.out.println();
		System.out.print("Initial outputs: [ ");
		for( final Value out : outs ) {
			System.out.print(out.data() + " ");
		}
		System.out.println("]");

		System.out.print("Target outputs: [ ");
		for( final double y : Ys ) {
			System.out.print(y + " ");
		}
		System.out.println("]");
		System.out.println();

		// training loop
		for( int i = 1; i <= 100; ++i ) {
			final Value[] outputs = new Value[4];
			for( int j = 0; j < Xs.length; ++j ) {
				outputs[j] = nn.forward(Xs[j])[0];
			}

			final MSELoss loss_func = new MSELoss(); // Mean Squared Error (MSE) loss function
			final Value loss = loss_func.loss(outputs, Ys);
			final SGD optim = new SGD(nn.parameters(), 0.1); // SGD optimizer

			if( i % 20 == 0 || i == 1 )
				System.out.println("iter: " + i + ", loss: " + loss.data());

			// backpropagation and optimization
			optim.zeroGrad();
			ValueUtil.backward(loss);
			optim.step();
		}

		// check newly trained outputs
		outs.clear();
		for( int i=0; i < Xs.length; ++i ) {
			outs.add(nn.forward(Xs[i])[0]);
		}

		System.out.println();
		System.out.print("Final outputs: [ ");
		for(Value out : outs) {
			System.out.print(out.data() + " ");
		}
		System.out.println("]");
		System.out.print("Target outputs: [ ");
		for(double y : Ys) {
			System.out.print(y + " ");
		}
		System.out.println("]");
	}
}
