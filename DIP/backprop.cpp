#include "stdafx.h"
#include "backprop.h"

#define LAMBDA 1.0
#define ETA 0.1

#define E 2.71828182845904523536

#define SQR( x ) ( ( x ) * ( x ) )

void randomize( double * p, int n ) 
{
	for ( int i = 0; i < n; i++ ) {
		p[i] = ( double )rand() / ( RAND_MAX );
	}
}

NN * createNN( int n, int h, int o ) 
{
	srand(time(NULL));
	NN * nn = new NN;
	
    nn->n = new int[3];
	nn->n[0] = n;
	nn->n[1] = h;
	nn->n[2] = o;
	nn->l = 3;

	nn->w = new double ** [nn->l - 1];
    

	for ( int k = 0; k < nn->l - 1; k++ ) 
    {
		nn->w[k] = new double * [nn->n[k + 1]];
		for ( int j = 0; j < nn->n[k + 1]; j++ ) 
        {
			nn->w[k][j] = new double[nn->n[k]];			
			randomize( nn->w[k][j], nn->n[k]);
			// BIAS
			//nn->w[k][j] = new double[nn->n[k] + 1];			
			//randomize( nn->w[k][j], nn->n[k] + 1 );
		}		
	}

	nn->y = new double * [nn->l];
	for ( int k = 0; k < nn->l; k++ ) {
		nn->y[k] = new double[nn->n[k]];
		memset( nn->y[k], 0, sizeof( double ) * nn->n[k] );
	}

	nn->in = nn->y[0];
	nn->out = nn->y[nn->l - 1];

	nn->d = new double * [nn->l];
	for ( int k = 0; k < nn->l; k++ ) {
		nn->d[k] = new double[nn->n[k]];
		memset( nn->d[k], 0, sizeof( double ) * nn->n[k] );
	}

	return nn;
}

void releaseNN( NN *& nn ) 
{
	for ( int k = 0; k < nn->l - 1; k++ ) {
		for ( int j = 0; j < nn->n[k + 1]; j++ ) {
			delete [] nn->w[k][j];
		}
		delete [] nn->w[k];
	}
	delete [] nn->w;
		
	for ( int k = 0; k < nn->l; k++ ) {
		delete [] nn->y[k];
	}
	delete [] nn->y;
	
	for ( int k = 0; k < nn->l; k++ ) {
		delete [] nn->d[k];
		
	}
	delete [] nn->d;

	delete [] nn->n;

	delete nn;
	nn = NULL;
}

void feedforward( NN * nn ) 
{ 
	int count = 0;

	//std::cout << std::endl;

	for (int layer = 1; layer < nn->l; layer++) {
		//double* input = nn->y[layer - 1]; // input

		for (int currentNeuron = 0; currentNeuron < nn->n[layer]; currentNeuron++) {
			double potential = 0;

			count++;

			for (int previousNeuron = 0; previousNeuron < nn->n[layer - 1]; previousNeuron++) {
				
				potential += nn->w[layer - 1][currentNeuron][previousNeuron] * nn->y[layer - 1][previousNeuron];

			}
			double sigmoid = 1. / (1. + pow(E, -LAMBDA * potential));

			nn->y[layer][currentNeuron] = sigmoid;

 			//std::cout << "k: " << layer << " i: " << currentNeuron << " y: " << nn->y[layer][currentNeuron] << " | ";
		}
			
		//std::cout << std::endl;
	}

	/*std::cout << count << std::endl;

	std::cout << "output: ";

	for (int i = 0; i < 2; i++)
		std::cout << nn->y[2][i] << ' ';

	std::cout << std::endl;*/
}

double backpropagation( NN * nn, double * t ) 
{
	double error = 0.0;

	const int outputLayer = nn->l - 1;
	const int outputLayerSize = nn->n[outputLayer];
	for (int i = 0; i < outputLayerSize; i++) {
		const double te = t[i] - nn->y[outputLayer][i];
		error += te * te;
	}

	// output layer error
	for (int i = 0; i < outputLayerSize; i++) {
		const double lamb = LAMBDA * nn->y[outputLayer][i] * (1. - nn->y[outputLayer][i]);
		const double e = (t[i] - nn->y[outputLayer][i]) * lamb;

		nn->d[outputLayer][i] = e;
	}

	// hidden layer error
	const int hiddenLayerCount = outputLayer - 1;
	//const int hiddenLayerSize = nn->n[hiddenLayer];
	for (int layer = 1; layer <= hiddenLayerCount; layer++) {
		for (int i = 0; i < nn->n[layer]; i++) {
			const double lamb = LAMBDA * nn->y[layer][i] * (1. - nn->y[layer][i]);
			double te = 0;

			for (int j = 0; j < nn->n[layer + 1]; j++) {
				te += nn->d[layer + 1][j] * nn->w[layer][j][i];
			}

			nn->d[layer][i] = te * lamb;
		}
	}

	// change weights for each connection/neruron
	for (int layer = 0; layer < nn->l - 1; layer++) {
		for (int i = 0; i < nn->n[layer + 1]; i++) {
			for (int j = 0; j < nn->n[layer]; j++) {
				nn->w[layer][i][j] += ETA * nn->d[layer + 1][i] * nn->y[layer][j];
			}
		}
	}

	return error * 0.5;
}

void setInput( NN * nn, double * in, bool verbose ) 
{
	memcpy( nn->in, in, sizeof( double ) * nn->n[0] );

	if ( verbose ) {
		printf( "input=(" );
		for ( int i = 0; i < nn->n[0]; i++ ) {
			printf( "%0.3f", nn->in[i] );
			if ( i < nn->n[0] - 1 ) {
				printf( ", " );
			}
		}
		printf( ")\n" );
	}
}

int getOutput( NN * nn, bool verbose ) 
{	
    double max = 0.0;
    int max_i = 0;
    if(verbose) printf( " output=" );
	for ( int i = 0; i < nn->n[nn->l - 1]; i++ ) 
    {
		if(verbose) printf( "%0.3f ", nn->out[i] );
        if(nn->out[i] > max) {
            max = nn->out[i];
            max_i = i;
        }
	}
	if(verbose) printf( " -> %d\n" , max_i);
    if(nn->out[0] > nn->out[1] && nn->out[0] - nn->out[1] < 0.1) return 2;
    return max_i;
}
