#import<fstream>
#define Q(x,n) for(z x=0;x<n;++x)
using f=float;using z=int;using V=void;
struct L{z c,s,k,t,ks,r;f*d,*w,*b,*a;z v,n;};
V init_layer(L&p,L&l){if(l.t<3){(l.t)?((l.t>1)?(l.s=p.s/2,l.c=p.c,l.n=p.c*l.s*l.s):(l.s=p.s-l.k+1,l.v=p.c*l.c*l.k*l.k,l.w=l.d,l.b=l.d+l.v,l.n=l.c*(p.s-l.k+1)*(p.s-l.k+1))):(l.s=1,l.v=p.c*p.s*p.s*l.c,l.w=l.d,l.b=l.d+l.v,l.n=l.c);l.a=new f[l.n];}else{l.n=l.c*l.s*l.s;}}
V inner_product(L&i,L&o){Q(oc,o.c){f s=0;for(z ic=0;ic<i.c*i.s*i.s;s+=i.a[ic]*o.w[oc*i.c*i.s*i.s+ic++]);f r=s+o.b[oc];o.a[oc]=o.r?(r>0?r:0):r;}}
V max_pool(L&i,L&o){z s=i.s,k=o.k;for(z j=0;j<o.c*o.s*o.s;o.a[j++]=-9e9);Q(j,i.c*s*s){f*p=o.a+(j/(s*s))*o.s*o.s+((j%(s*s)/s)/k)*o.s+(j%(s*s))%s/k,a;for(z y=0;y<k;a=i.a[j+(y/k)*s+y++%k],*p=a>*p?a:*p);}}
V convolution(L&i,L&o){
    z k=o.k,s=o.s;
    Q(c,o.c){Q(h,s){Q(w,s){
        f E=0;
        Q(q,i.c){Q(y,k){Q(x,k){
            	       E += i.a[q*i.s*i.s + (h+y)*i.s + w + x]
            	    		  * o.w[c*i.c*k*k + q*k*k + y*k + x];
	   }}}
        o.a[c*s*s + h*s + w] = E + o.b[c];
    }}}}
V read_binary_f(f*d,char*n,z l){std::ifstream fin(n,std::ios::binary);fin.read(reinterpret_cast<char*>(d),l*4);}
main(z,char**c){
    f*d=new f[784],*w=new f[431080];read_binary_f(d,c[1],784);read_binary_f(w,c[2],431080);
    L n[7]={{1,28,0,3,0,0,0,0,0,d},{20,0,5,1,0,0,w},{0,0,2,2},{50,0,5,1,0,0,w+520},{0,0,2,2},{500,0,0,0,0,1,w+25570},{10,0,0,0,0,0,w+426070}},*l=0,*p=n;
    for(;p-n<7;init_layer(*l,*p),l=p++);
    V(*F[3])(L&,L&){inner_product, convolution, max_pool};for(p=n,l=p++;p-n<7;F[p->t](*l,*p),l=p++);
    d=l->a,w=d;for(;d-l->a<l->c;*d>*w?w=d:0,++d);return 10+w-d;}
