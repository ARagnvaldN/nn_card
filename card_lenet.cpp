#import<stdio.h>
#define Q(x,n) for(z x=0;x<n;++x){
#define D [](L&i,L&o)
using f=float;using z=int;
struct L{z c,s,k,t,r;f*d,*w,*b,*a;};
void R(f*d,char*n,z l){fread((char*)d,4,l,fopen(n,"r"));}
main(z q,char**c){q=431080;
f*d=new f[784],*w=new f[q];R(d,c[1],784);R(w,c[2],q);
L n[7]={{1,28,0,3,0,0,0,0,d},{20,0,5,1,0,w},{0,0,2,2},{50,0,5,1,0,w+520},{0,0,2,2},{500,0,0,0,1,w+25570},{10,0,0,0,0,w+426070}},
*l=0,*p=n;for(;p-n<7;D{z n,v;if(o.t<3){(o.t)?((o.t>1)?(o.s=i.s/2,o.c=i.c,n=i.c*o.s*o.s):(o.s=i.s-o.k+1,v=i.c*o.c*o.k*o.k,o.w=o.d,o.b=o.d+v,n=o.c*(i.s-o.k+1)*(i.s-o.k+1))):(o.s=1,v=i.c*i.s*i.s*o.c,o.w=o.d,o.b=o.d+v,n=o.c);o.a=new f[n];}else{n=o.c*o.s*o.s;}}(*l,*p),l=p++);
void(*F[3])(L&,L&){
D{Q(c,o.c)f s=0;Q(q,i.c*i.s*i.s)s+=i.a[q]*o.w[c*i.c*i.s*i.s+q];};f r=s+o.b[c];o.a[c]=o.r?(r>0?r:0):r;}},
D{z k=o.k,s=o.s;Q(c,o.c)Q(h,s)Q(w,s)f E=0;Q(q,i.c)Q(y,k)Q(x,k)E+=i.a[q*i.s*i.s+(h+y)*i.s+w+x]*o.w[c*i.c*k*k+q*k*k+y*k+x];}}}o.a[c*s*s+h*s+w]=E+o.b[c];}}}},
D{z s=i.s,l=o.s,k=o.k;Q(c,i.c)Q(h,l)Q(w,l)f*a=i.a+c*s*s+h*2*s+w*2,*b=a+s;b=*b>b[1]?b:b+1;a=*a>a[1]?a:a+1;o.a[c*l*l+h*l+w]=*a>*b?*a:*b;}}}}};
for(p=n,l=p++;p-n<7;F[p->t](*l,*p),l=p++);
w=l->a;d=w;Q(i,l->c)w[i]>=*d?q=i,d=w+i:0;};return q;}
