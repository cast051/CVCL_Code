__kernel void Image_Remap(__read_only  image2d_t InputImage,__read_only image2d_t Mapx_Mat ,__read_only image2d_t Mapy_Mat ,__write_only image2d_t OutputImage,sampler_t sampler)
{ 
	const int x=get_global_id(0);
	const int y=get_global_id(1);

	// const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
	//							 CLK_ADDRESS_CLAMP 	| 
	//							 CLK_FILTER_LINEAR;
	int2 coorf=(int2)(x, y);
	
	float4 Mapxdata=read_imagef(Mapx_Mat,sampler,coorf);
	float4 Mapydata=read_imagef(Mapy_Mat,sampler,coorf);
	

	int Mapx_down = (int)floor(Mapxdata[0]);
	int Mapx_up = Mapx_down+1;
	int Mapy_down = (int)floor(Mapydata[0]);
	int Mapy_up = Mapy_down+1;

	float map_weight00 = (float)(Mapx_up - Mapxdata[0])*(Mapy_up - Mapydata[0]);
	float map_weight01 = (float)(Mapxdata[0] - Mapx_down)*(Mapy_up - Mapydata[0]);
	float map_weight10 = (float)(Mapx_up - Mapxdata[0])*(Mapydata[0] - Mapy_down);
	float map_weight11 = (float)(Mapxdata[0] - Mapx_down)*(Mapydata[0] - Mapy_down);

	float4 map_weight00_f4=(float4)(map_weight00,map_weight00,map_weight00,map_weight00);
	float4 map_weight01_f4=(float4)(map_weight01,map_weight01,map_weight01,map_weight01);
	float4 map_weight10_f4=(float4)(map_weight10,map_weight10,map_weight10,map_weight10);
	float4 map_weight11_f4=(float4)(map_weight11,map_weight11,map_weight11,map_weight11);

	
	int4 InputImg00=read_imagei(InputImage,sampler,(int2)(Mapx_down,Mapy_down));
	int4 InputImg01=read_imagei(InputImage,sampler,(int2)(Mapx_up,Mapy_down));
	int4 InputImg10=read_imagei(InputImage,sampler,(int2)(Mapx_down,Mapy_up));
	int4 InputImg11=read_imagei(InputImage,sampler,(int2)(Mapx_up,Mapy_up));

	float4 InputImg00_f4=convert_float4(InputImg00);
	float4 InputImg01_f4=convert_float4(InputImg01);
	float4 InputImg10_f4=convert_float4(InputImg10);
	float4 InputImg11_f4=convert_float4(InputImg11);


	float4 value= (float4)(map_weight00_f4 * InputImg00_f4\
					     + map_weight01_f4 * InputImg01_f4\
				  	     + map_weight10_f4 * InputImg10_f4\
					   	 + map_weight11_f4 * InputImg11_f4);
	
	int4 value_int4=convert_int4(value);

	write_imagei(OutputImage,coorf,value_int4);
}
