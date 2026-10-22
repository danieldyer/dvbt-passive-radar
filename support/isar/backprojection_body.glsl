layout(local_size_x=64) in;
// Geometry is relative to the target centre, never float ECEF coordinates.
// Ten vec4s per row: centre-T, centre-R, body axes, centre/body-axis velocities,
// then (cropped profile coordinate, cos(k L), sin(k L), information).
layout(set=0,binding=0,std430) readonly buffer Geometry {vec4 geometry[];};
layout(set=0,binding=1,std430) readonly buffer Profiles {vec2 profile[];};
layout(set=0,binding=2,std430) readonly buffer Pixels {vec4 offset[];};
layout(set=0,binding=3,std430) readonly buffer Interpolation {float coefficient[];};
layout(set=0,binding=4,std430) writeonly buffer Partial {vec4 partial[];};
layout(set=0,binding=6,std430) readonly buffer DenseProfiles {vec4 dense[];};
layout(push_constant) uniform Parameters {
    uint rows; uint ranges; uint pixels; uint phases;
    uint block_rows; uint blocks; float k; float symbol_duration;
    uint oversample;
} p;
vec2 multiply(vec2 a,vec2 b){return vec2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
void main(){
    uint pixel=gl_GlobalInvocationID.x,block=gl_WorkGroupID.y;
    if(pixel>=p.pixels)return;
    vec3 point=offset[pixel].xyz;
    vec2 sum=vec2(0);float information=0,squared_information=0;
    #ifdef CHANNEL_UPDATE
    uint first_row=0,last_row=p.rows;
#else
    uint first_row=block*p.block_rows,last_row=min((block+1)*p.block_rows,p.rows);
#endif
    for(uint row=first_row;row<last_row;++row){
        uint index=10*row;
        vec4 g=geometry[index+9];
        if(g.w<=0)continue;
        float sign=1.;
#ifdef CHANNEL_UPDATE
        if(uint(geometry[index+6].w)!=block)continue;
        sign=geometry[index+7].w;
#endif
        float k=geometry[index+5].w>0 ? geometry[index+5].w : p.k;
        float gain_power=geometry[index+8].w>0 ? geometry[index+8].w : 1.;
        vec3 d=geometry[index+2].xyz*point.x+geometry[index+3].xyz*point.y+geometry[index+4].xyz*point.z;
        vec4 tx=geometry[index],rx=geometry[index+1];
        float a=length(tx.xyz+d),b=length(rx.xyz+d);
        // Difference of norms, rationalised to avoid subtracting ~40 km floats.
        float dd=dot(d,d);
        float delta=(2*dot(tx.xyz,d)+dd)/(a+tx.w)+(2*dot(rx.xyz,d)+dd)/(b+rx.w);
        float coordinate=g.x+delta*(8000000./299792458.);
        int base=int(floor(coordinate));
        if(base<31 || base+32>=int(p.ranges))continue;
        float fraction=fract(coordinate)*p.phases;
        uint phase=min(uint(fraction),p.phases-1);float mix_fraction=fraction-phase;
        vec2 zero=vec2(0),one=vec2(0);
        if(p.oversample>0){
            uint stride=(p.ranges-63)*p.oversample;
            float q=(coordinate-31)*p.oversample;
            uint at=uint(floor(q));
            // Both neighboring dense samples have the full original 64 taps.
            if(at+1>=stride)continue;
            vec4 z=mix(dense[row*stride+at],dense[row*stride+at+1],fract(q));
            zero=z.xy;one=z.zw;
        }else for(int tap=0;tap<64;++tap){
            float c=mix(coefficient[phase*64+tap],coefficient[(phase+1)*64+tap],mix_fraction);
            uint address=row*p.ranges+uint(base+tap-31);
            zero+=c*profile[address];one+=c*profile[p.rows*p.ranges+address];
        }
        vec3 v=geometry[index+5].xyz+geometry[index+6].xyz*point.x+geometry[index+7].xyz*point.y+geometry[index+8].xyz*point.z;
        // -j 2 pi fd T C1, with fd= -k/(2 pi) dL/dt.
        float omega=-k*dot((tx.xyz+d)/a+(rx.xyz+d)/b,v)*p.symbol_duration;
        vec2 z=zero+omega*vec2(one.y,-one.x);
        float angle=k*delta;
        sum+=sign*g.w*multiply(multiply(z,g.yz),vec2(cos(angle),sin(angle)));
        information+=sign*g.w*gain_power;squared_information+=g.w*g.w*gain_power*gain_power;
    }
    partial[block*p.pixels+pixel]=vec4(sum,information,squared_information);
}
