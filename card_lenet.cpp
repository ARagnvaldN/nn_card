#import<fstream>
#define Q(x,n) for(z x=0;x<n;++x)
using f=float;using z=int;using V=void;
struct L{z c,s,k,t,r;f*d,*w,*b,*a;};
V X(L&i,L&o){Q(c,o.c){f s=0;Q(q,i.c*i.s*i.s){s+=i.a[q]*o.w[c*i.c*i.s*i.s+q];};f r=s+o.b[c];o.a[c]=o.r?(r>0?r:0):r;}}
V M(L&i,L&o){z s=i.s,k=o.k;Q(j,o.c*o.s*o.s){o.a[j]=-9e9;};Q(j,i.c*s*s){f*p=o.a+(j/(s*s))*o.s*o.s+((j%(s*s)/s)/k)*o.s+(j%(s*s))%s/k,a;Q(y,k){a=i.a[j+(y/k)*s+y%k];*p=a>*p?a:*p;};}}
V C(L&i,L&o){z k=o.k,s=o.s;Q(c,o.c){Q(h,s){Q(w,s){f E=0;Q(q,i.c){Q(y,k){Q(x,k){E+=i.a[q*i.s*i.s+(h+y)*i.s+w+x]*o.w[c*i.c*k*k+q*k*k+y*k+x];}}}o.a[c*s*s+h*s+w]=E+o.b[c];}}}}
V R(f*d,char*n,z l){std::ifstream fin(n,std::ios::binary);fin.read((char*)d,l*4);}
main(z q,char**c){q=431080;f*d=new f[784],*w=new f[q];R(d,c[1],784);R(w,c[2],q);
L n[7]={{1,28,0,3,0,0,0,0,d},{20,0,5,1,0,w},{0,0,2,2},{50,0,5,1,0,w+520},{0,0,2,2},{500,0,0,0,1,w+25570},{10,0,0,0,0,w+426070}},
*l=0,*p=n;for(;p-n<7;[](L&p,L&l){z n,v;if(l.t<3){(l.t)?((l.t>1)?(l.s=p.s/2,l.c=p.c,n=p.c*l.s*l.s):(l.s=p.s-l.k+1,v=p.c*l.c*l.k*l.k,l.w=l.d,l.b=l.d+v,n=l.c*(p.s-l.k+1)*(p.s-l.k+1))):(l.s=1,v=p.c*p.s*p.s*l.c,l.w=l.d,l.b=l.d+v,n=l.c);l.a=new f[n];}else{n=l.c*l.s*l.s;}}(*l,*p),l=p++);
V(*F[3])(L&,L&){X,C,M};for(p=n,l=p++;p-n<7;F[p->t](*l,*p),l=p++);
w=l->a;d=w;Q(i,l->c){w[i]>=*d?q=i,d=w+i:0;};return q;}//
