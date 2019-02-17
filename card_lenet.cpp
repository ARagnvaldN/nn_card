#import<fstream>
using f=float;using z=int;using V=void;
struct L{z c,s,k,t,ks,r;f*d,*w,*b,*a;z nw,na;};
V init_layer(L*p,L*l){if(l->t<3){(l->t)?((l->t>1)?(l->s=p->s/2,l->c=p->c,l->na=p->c*l->s*l->s):(l->s=p->s-l->k+1,l->nw=p->c*l->c*l->k*l->k,l->w=l->d,l->b=l->d+l->nw,l->na=l->c*(p->s-l->k+1)*(p->s-l->k+1))):(l->s=1,l->nw=p->c*p->s*p->s*l->c,l->w=l->d,l->b=l->d+l->nw,l->na=l->c);l->a=new f[l->na];}else{l->na=l->c*l->s*l->s;l->a=new f[l->na];std::copy(l->d,l->d+l->na,l->a);}}
V inner_product(L*i,L*o){for(z oc=0;oc<o->c;++oc){f s=0;for(z ic=0;ic<i->c*i->s*i->s;s+=i->a[ic]*o->w[oc*i->c*i->s*i->s+ic++]);f r=s+o->b[oc];o->a[oc]=o->r?(r>0?r:0):r;}}
V max_pool(L*i,L*o){z s=i->s,k=o->k;for(z j=0;j<o->c*o->s*o->s;o->a[j++]=-9e9);for(z j=0;j<i->c*s*s;++j){f*p=o->a+(j/(s*s))*o->s*o->s+((j%(s*s)/s)/k)*o->s+(j%(s*s))%s/k,a;for(z y=0;y<k;a=i->a[j+(y/k)*s+y++%k],*p=a>*p?a:*p);}}
V convolution(L*i,L*o){
    z k=o->k;
    for(z c_outer=0;c_outer<o->c;++c_outer){
        for(z h_outer=0;h_outer<o->s;++h_outer){
        	  for(z w_outer=0;w_outer<o->s;++w_outer){
        
        		f E=0;
        		for (z c_inner = 0; c_inner < i->c; ++c_inner) {
    
      		    for (z kernel_y = 0; kernel_y < k; ++kernel_y) {
    				   for (z kernel_x = 0; kernel_x < k; ++kernel_x) {
    			    	    
    			    	       E += i->a[c_inner * i->s * i->s
    			    	    		               	    + (h_outer + kernel_y) * i->s
    			    	    						    + w_outer + kernel_x]
    			    	    		  * o->w[c_outer * i->c
    			    	    		  					   * k*k
    			    	    		  				+ c_inner * k*k
    			    	    		  				+ kernel_y * k
    			    	    		  				+ kernel_x];
   			        }
      		    }
			}
               o->a[c_outer * o->s * o->s
                                   + h_outer * o->s
                   	     	     + w_outer] = E + o->b[c_outer];
		  }
	   }
    }
    
}
V read_binary_f(f*d,char*n,z l){std::ifstream fin(n,std::ios::binary);fin.read(reinterpret_cast<char*>(d),l*4);}
main(z c,char**a){
    z i=0,g=0,x=6;
    f*d=new f[784],*w=new f[431080];read_binary_f(d,a[1],784);read_binary_f(w,a[2],431080);
    L n[7]={{1,28,0,3,0,0,d},{20,0,5,1,0,0,w},{0,0,2,2},{50,0,5,1,0,0,w+520},{0,0,2,2},{500,0,0,0,0,1,w+25570},{10,0,0,0,0,0,w+426070}};
    L*l=0;for(i=0;i<7;init_layer(l,&n[i]),l=&n[i++]);
    for(i=0;i<7-1;++i){n[i+1].t?n[i+1].t==1?convolution(&n[i],&n[i+1]):max_pool(&n[i],&n[i+1]):inner_product(&n[i],&n[i+1]);}
    f m=*n[x].a;
    for(i=0;i<n[x].c;n[x].a[i]>m?g=i,m=n[x].a[i]:0,++i);
    return g;}
