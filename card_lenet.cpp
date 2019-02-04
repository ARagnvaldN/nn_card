#include <fstream>
#define N 7
struct L{int c,s,k,t,z;bool r;float*d,*w,*b,*a;int nw,na;};
void init_layer(L*p,L*l){if(l->t<3){(l->t)?((l->t>1)?(l->s=p->s/2,l->c=p->c,l->na=p->c*l->s*l->s):(l->s=p->s-l->k+1,l->nw = p->c*l->c*l->k*l->k,l->w=l->d,l->b=l->d+l->nw,l->na=l->c*(p->s-l->k+1)*(p->s-l->k+1))):(l->s=1,l->nw=p->c*p->s*p->s*l->c,l->w=l->d,l->b=l->d+l->nw,l->na=l->c);l->a=new float [l->na];for(int i=0;i<l->na;++i){l->a[i]=0;}}else{l->na=l->c*l->s*l->s;l->a=new float [l->na];std::copy(l->d,l->d+l->na,l->a);}}
void inner_product(L*i,L*o){
    for(int oc=0;oc<o->c;++oc){
	   float s=0;
	   for (int ic=0;ic<i->c;++ic){
		  for(int y=0;y<i->s;++y){
			 for(int x=0;x<i->s;++x){

			 s += i->a[ic * i->s * i->s
								  + y * i->s
								  + x]
				    * o->w[oc * i->c * i->s * i->s
				    				  + ic * i->s * i->s
								  + y * i->s
								  + x];
	   		}
		  }
	   }
	   float r=s+o->b[oc];
	   if(o->r){if(r>0)o->a[oc]=r;
	   }else{o->a[oc]=s+o->b[oc];}
    }}
void max_pool(L*i,L*o){
    for (int c_out = 0; c_out < o->c; ++c_out) {
	   for (int h_out = 0; h_out < o->s; ++h_out) {
		  for (int w_out = 0; w_out < o->s; ++w_out) {
			 float max = -10000.0f;
			 for (int y_kernel = 0; y_kernel < o->k; ++y_kernel) {
				for (int x_kernel = 0; x_kernel < o->k; ++x_kernel) {
				    float current_val = i->a[c_out * i->s
					   								    * i->s
								   				   + (h_out * o->k
												      + y_kernel)
												   		 * i->s
								   				   + w_out * o->k
												   + x_kernel];
				    if (current_val > max)
					   max = current_val;
				}
			 }
			 o->a[c_out * o->s * o->s
							 + h_out * o->s
							 + w_out] = max;
		  }
	   }
    }
}

void convolution(L*i,L*o){
    int k = o->k;
    for (int c_outer = 0; c_outer < o->c; ++c_outer) {
        for (int h_outer = 0; h_outer < o->s; ++h_outer) {
        	  for (int w_outer = 0; w_outer < o->s; ++w_outer) {
        
        		float sum = 0.f;
        		for (int c_inner = 0; c_inner < i->c; ++c_inner) {
    
      		    for (int kernel_y = 0; kernel_y < k; ++kernel_y) {
    				   for (int kernel_x = 0; kernel_x < k; ++kernel_x) {
    			    	    
    			    	       sum += i->a[c_inner * i->s * i->s
    			    	    		               	    + (h_outer + kernel_y) * i->s
    			    	    						    + w_outer + kernel_x]
    			    	    		  * o->w[c_outer * i->c
    			    	    		  					   * k * k
    			    	    		  				+ c_inner * k * k
    			    	    		  				+ kernel_y * k
    			    	    		  				+ kernel_x];
   			        }
      		    }
			}
               o->a[c_outer * o->s * o->s
                                   + h_outer * o->s
                   	     	     + w_outer] = sum + o->b[c_outer];
		  }
	   }
    }
    
}
void forward(L*l,L*c){c->t?c->t==1?convolution(l,c):max_pool(l,c):inner_product(l,c);}
void read_binary_float(float*d,char*f,int l){std::ifstream fin(f,std::ios::binary);fin.read(reinterpret_cast<char*>(d),l*4);}
int main(int c,char**a){
    float*d=new float[784];read_binary_float(d,a[1],784);
    float * w = new float[431080];read_binary_float(w,a[2],431080);
    L n[N]={{1,28,0,3,0,0,d},{20,0,5,1,0,0,w},{0,0,2,2},{50,0,5,1,0,0,w+520},{0,0,2,2},{500,0,0,0,0,1,w+25570},{10,0,0,0,0,0,w+426070}};
    L*l=0;for(int i=0;i<N;++i){init_layer(l,&n[i]);l=&n[i];}
    for(int i=0;i<N-1;++i){n[i+1].t?n[i+1].t==1?convolution(&n[i],&n[i+1]):max_pool(&n[i],&n[i+1]):inner_product(&n[i],&n[i+1]);}
    int idx=N-1,am=0;
    float max=n[idx].a[0];
    for(int i=0;i<n[idx].c;++i){n[idx].a[i]>max?am=i,max=n[idx].a[i]:0;}
    return am;}
