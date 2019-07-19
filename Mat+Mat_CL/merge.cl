__kernel void Image_Merge(__read_only  image2d_t InputImage1,__read_only image2d_t InputImage2 ,__write_only image2d_t OutputImage,sampler_t sampler)
{ 
	const int x=get_global_id(0);
	const int y=get_global_id(1);

	
	// const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
	//							 CLK_ADDRESS_CLAMP 	| 
	//							 CLK_FILTER_LINEAR;

	int2 coorf=(int2)(x, y);
	int4 value =(read_imagei(InputImage1,sampler,coorf)+read_imagei(InputImage2,sampler,coorf))/2;
	
	write_imagei(OutputImage,coorf,value);

}
