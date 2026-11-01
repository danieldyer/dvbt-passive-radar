// Both global and workgroup-cached kernels use this same local-linear fit.
// pilot_observation(row,j) supplies the aligned measured pilot, never data.
vec2 pilot_fit(uint s,uint j,uint support,bool continual){
    int begin=max(0,int(s)-int(support)),end=min(int(pc.count)-1,int(s)+int(support));
    uint mod=floatBitsToUint(source_pilots[rows[begin]*pc.stride].x);
    int first=continual ? begin : begin+int((j%4+4-mod)%4);
    int step=continual ? 1 : 4;
    // The pilot epochs are an arithmetic progression. Compute their moments
    // exactly in integer arithmetic once, not alongside every complex sum.
    // With support <=64 these intermediates fit comfortably in signed int.
    int count=first<=end ? (end-first)/step+1 : 0,a=first-int(s);
    int pairs=count*(count-1);
    float n=float(count),t=float(count*a+step*pairs/2);
    float tt=float(count*a*a+a*step*pairs+step*step*pairs*(2*count-1)/6);
    vec2 y=vec2(0),ty=vec2(0);
    for(int r=first;r<=end;r+=step){
        float dt=float(r)-float(s);vec2 z=pilot_observation(r,j);
        y+=z;ty+=dt*z;
    }
    float determinant=n*tt-t*t;
    return determinant>0 ? (tt*y-t*ty)/determinant : y/n;
}
