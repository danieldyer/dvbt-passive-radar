#include "soft_quantizer.glsl"
// Exact max-log axis distances. Quantised likelihood +7, positive favours zero.
vec3 axis_likelihood(float x,float scale,uint bits){
    if(bits==2)return vec3(2.8284271247461903*x*scale,0,0);
    float energy=bits==4 ? 10 : 42,u=abs(x)*sqrt(energy);
    float a=bits==4 ? (u<2 ? 1 : 3) : (u<2 ? 1 : u<4 ? 3 : u<6 ? 5 : 7);
    float sign_llr=(a+1)*(2*u+1-a)*scale/energy;
    sign_llr=x<0 ? -sign_llr : sign_llr;
    if(bits==4)return vec3(sign_llr,4*(u-2)*scale/energy,0);
    float inner=u>2 ? 3 : 1,outer=u>6 ? 7 : 5;
    float zero=u>4 ? 7 : 1,one=u>4 ? 5 : 3;
    return vec3(sign_llr,(outer-inner)*(2*u-inner-outer)*scale/energy,(zero-one)*(2*u-zero-one)*scale/energy);
}
uint quantized_likelihood(float x){
    if(isnan(x))return 7;
    float y=clamp(x,-7,7),a=abs(y);int q=int(a);q+=int(a-float(q)>=.5);return uint((y<0 ? -q : q)+7);
}
uint soft_qam(vec2 z,float scale,uint bits,bool minimax){
    if(any(isnan(z))||any(isinf(z))||isnan(scale)||isinf(scale))return 0x777777;
    vec3 re=axis_likelihood(z.x,scale,bits),im=axis_likelihood(z.y,scale,bits);
    uint packed=0;
    for(uint j=0;j<bits/2;++j){packed|=(minimax ? minimax_likelihood(re[j]) : quantized_likelihood(re[j]))<<(8*j);packed|=(minimax ? minimax_likelihood(im[j]) : quantized_likelihood(im[j]))<<(8*j+4);}
    return packed;
}
uint soft_qam(vec2 z,float scale,uint bits){return soft_qam(z,scale,bits,false);}
