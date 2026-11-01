uint hard_qam(vec2 z,uint qam_bits){
    if(any(isnan(z)) || any(isinf(z)))return 255;
    uint axis=qam_bits/2,axis_half=1U<<(axis-1),result=0;
    float scale=qam_bits==2 ? 1.4142135381698608 : qam_bits==4 ? 3.1622776985168457 : 6.480740547180176;
    for(uint a=0;a<2;++a){
        precise float magnitude=abs(z[a])*scale*.5;
        uint rank=uint(clamp(float(axis_half)-magnitude,0.,float(axis_half-1))),gray=rank^(rank>>1);
        // Some devices flush subnormal arithmetic to zero. Preserve its sign
        // decision while treating both signed zeros exactly like the CPU.
        uint representation=floatBitsToUint(z[a]);
        result|=((representation>>31)*uint((representation&0x7fffffffU)!=0))<<a;
        for(uint j=0;j<axis-1;++j)result|=((gray>>(axis-2-j))&1U)<<(2+2*j+a);
    }
    return result;
}
