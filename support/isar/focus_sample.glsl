layout(set=0,binding=0,std430) readonly buffer Geometry {vec4 geometry[];};
layout(set=0,binding=1,std430) readonly buffer Coefficients {vec2 coefficient[];};
layout(set=0,binding=2,std430) readonly buffer Pixels {vec4 pixel[];};
layout(set=0,binding=6,std430) buffer Status {uint unsupported;};
layout(push_constant) uniform Parameters {
    uint stride;uint bands;uint cells;uint degree;uint pixels;uint blocks;
    int first_cell;uint block_rows;float symbol_duration;uint derivatives;uint derivative_mode;
} p;
vec2 multiply(vec2 a,vec2 b){return vec2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
vec2 times_i(vec2 a){return vec2(-a.y,a.x);}
void polynomial(uint row,uint cell,float x,out vec2 value,out vec2 derivative){
    uint first=(row*p.cells+cell)*(p.degree+1);
    vec2 a=vec2(0),b=a,da=a,db=a;
    for(int k=int(p.degree);k>0;--k){
        vec2 next=coefficient[first+uint(k)]+2*x*a-b;
        vec2 dn=2*a+2*x*da-db;
        b=a;a=next;db=da;da=dn;
    }
    value=coefficient[first]+x*a-b;derivative=2*(a+x*da-db);
}
void sample_row(uint row,vec3 point,out vec2 value_and_derivative[9]){
    for(uint j=0;j<p.derivatives+1;++j)value_and_derivative[j]=vec2(0);
        uint base=(10+8*p.derivatives)*row;vec4 g=geometry[base+9];if(g.w<=0)return;
        vec3 d=geometry[base+2].xyz*point.x+geometry[base+3].xyz*point.y+geometry[base+4].xyz*point.z;
        vec4 tx=geometry[base],rx=geometry[base+1];
        vec3 t=tx.xyz+d,r=rx.xyz+d;float a=length(t),b=length(r),dd=dot(d,d);
        float delta=(2*dot(tx.xyz,d)+dd)/(a+tx.w)+(2*dot(rx.xyz,d)+dd)/(b+rx.w);
        float coordinate=g.x+delta*(8000000./299792458.);int cell=int(floor(coordinate))-p.first_cell;
        if(cell<0 || cell>=int(p.cells)){atomicAdd(unsupported,1);return;}
        vec2 value,derivative;polynomial(row,uint(cell),2*fract(coordinate)-1,value,derivative);
        float k=geometry[base+5].w,angle=k*delta;
        vec2 phase=multiply(g.yz,vec2(cos(angle),sin(angle)));
        vec3 velocity=geometry[base+5].xyz+geometry[base+6].xyz*point.x+geometry[base+7].xyz*point.y+geometry[base+8].xyz*point.z;
        vec3 ut=t/a,ur=r/b,gradient=ut+ur;
        float omega=-k*p.symbol_duration*dot(gradient,velocity),m0,dm0;
        if(abs(omega)<.01){float w2=omega*omega;m0=1-w2/24+w2*w2/1920;dm0=-omega/12+omega*w2/480;}
        else{float x=omega/2;m0=sin(x)/x;dm0=.5*(x*cos(x)-sin(x))/(x*x);}
        value_and_derivative[0]=multiply(phase,value*m0);
        if(p.derivative_mode==0)return;
        vec3 gradient_rate=(velocity-ut*dot(ut,velocity))/a+(velocity-ur*dot(ur,velocity))/b;
        for(uint j=0;j<p.derivative_mode;++j){
            uint first=base+10+8*j;
            vec3 dp=geometry[first].xyz+geometry[first+2].xyz*point.x+geometry[first+3].xyz*point.y+geometry[first+4].xyz*point.z;
            vec3 dv=geometry[first+1].xyz+geometry[first+5].xyz*point.x+geometry[first+6].xyz*point.y+geometry[first+7].xyz*point.z;
            float path=dot(gradient,dp),carrier_path=path;
            float doppler=-k*p.symbol_duration*(dot(gradient,dv)+dot(gradient_rate,dp));
            if(geometry[first].w==1 || geometry[first].w==2){
                vec3 unit=geometry[first].w==1 ? ut : ur;
                float distance=geometry[first].w==1 ? a : b;
                vec3 baseline=geometry[first].w==1 ? -geometry[first+1].xyz : geometry[first+1].xyz;
                path=dot(-unit+baseline,dp);
                carrier_path=path-geometry[first+1].w;
                doppler=k*p.symbol_duration*dot((velocity-unit*dot(unit,velocity))/distance,dp);
            }
            vec2 change=derivative*(8000000./299792458.*path*m0)+value*(dm0*doppler)+times_i(value)*(k*carrier_path*m0);
            value_and_derivative[j+1]=multiply(phase,change);
        }
}
