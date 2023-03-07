#include <iostream>
#include <cstdio>
#include <cstring>
#include "jpeglib.h"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

namespace py = pybind11;

py::bytes jpeg_encode(py::array_t<uint8_t> img_arr, int quality)
{
  auto img = img_arr.request();
  int height=img.shape[0], width=img.shape[1];
  uint8_t *enc_buf=NULL;
  size_t enc_length=0;

  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  JSAMPROW row_pointer[1];      /* pointer to JSAMPLE row[s] */
  int row_stride;               /* physical row width in image buffer */

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_mem_dest(&cinfo, &enc_buf, &enc_length);

  cinfo.image_width = width;      /* image width and height, in pixels */
  cinfo.image_height = height;
  cinfo.input_components = 3;           /* # of color components per pixel */
  cinfo.in_color_space = JCS_RGB;       /* colorspace of input image */
  
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE /* limit to baseline-JPEG values */);
  jpeg_start_compress(&cinfo, TRUE);

  row_stride = width * 3; /* JSAMPLEs per row in image_buffer */
  JSAMPLE *img_buf = static_cast<JSAMPLE *>(img.ptr);
  while (cinfo.next_scanline < cinfo.image_height) {
    row_pointer[0] = & img_buf[cinfo.next_scanline * row_stride];
    (void) jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);

  py::capsule free_when_done(enc_buf, [](void *f) {
      // std::cerr << "free buffer" << std::endl;
      auto *foo = reinterpret_cast<uint8_t *>(f);
      free(foo);
  });

  return py::bytes((char *)enc_buf, enc_length);

  // return py::array_t<uint8_t>(
  //     {enc_length, }, // shape
  //     {sizeof(uint8_t)}, // C-style contiguous strides for double
  //     enc_buf, // the data pointer
  //     free_when_done); // numpy array references this parent
}


py::array_t<uint8_t> jpeg_decode(py::bytes bytes) {
  const auto bstr = static_cast<std::string>(bytes);
  auto *inbuf = (unsigned char *)(bstr.c_str());
  
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jerr.error_exit = [](j_common_ptr cinfo){
    char pszErr[1024];
    (cinfo->err->format_message)(cinfo, pszErr);
    jpeg_destroy_decompress((struct jpeg_decompress_struct *)cinfo);
    throw std::runtime_error(pszErr);
  };
  
  JSAMPLE *buffer;
  unsigned long row_stride, width, height, channel;
  
  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, inbuf, bstr.length());

  (void) jpeg_read_header(&cinfo, TRUE);
  (void) jpeg_start_decompress(&cinfo);

  height = cinfo.output_height;
  width = cinfo.output_width;
  channel = cinfo.output_components;
  row_stride = width * channel;

  buffer = (JSAMPLE *)malloc(sizeof(uint8_t)*height*width*channel);
  JSAMPARRAY line_buf = (*cinfo.mem->alloc_sarray)
    ((j_common_ptr) &cinfo, JPOOL_IMAGE, width*3, 1);

  for (unsigned int i=0; cinfo.output_scanline < cinfo.output_height; i++) {
    auto offset = cinfo.output_scanline * row_stride;
    jpeg_read_scanlines(&cinfo, line_buf, 1);
    memcpy(buffer+offset, line_buf[0], row_stride);
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  
  py::capsule free_when_done(buffer, [](void *f) {
      auto *foo = reinterpret_cast<uint8_t *>(f);
      free(foo);
  });

  return py::array_t<uint8_t>(
      {height, width, channel},
      {width*channel, channel, 1}, 
      buffer, 
      free_when_done); 
}


PYBIND11_PLUGIN(_jpegpy) {
  py::module m("_jpegpy", "libjpeg-turbo encode and decode");
  m.def("encode", &jpeg_encode, "encode numpy array to jpeg bytes");
  m.def("decode", &jpeg_decode, "decode numpy array to jpeg bytes");
  return m.ptr();
}

// vim: ts=2 sts=2 sw=2 expandtab
