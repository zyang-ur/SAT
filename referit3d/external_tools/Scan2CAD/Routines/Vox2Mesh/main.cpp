#include <cmath>
#include <fstream>
#include <random>
#include <chrono>
#include <unordered_map>
#include <iostream>

#include <eigen3/Eigen/Dense>
#include "Box3D.h"
#include "LoaderVOX.h"
#include "LoaderMesh.h"
#include "SE3.h"
#include "Colormap.h"
#include "args.hxx"

struct InputArgs {
	std::string in;
	std::string out;
	bool is_unitless;
	bool redcenter;
	std::string cmap;
	float trunc;
} inargs;


Eigen::MatrixXf box_vertices, box_normals;
Eigen::Matrix<uint32_t, -1, -1> box_elements;

struct PlyMesh {
	Eigen::Matrix<float, -1, -1> V;
	Eigen::Matrix<float, -1, -1> N;
	Eigen::Matrix<uint8_t, -1, -1> C;
	Eigen::Matrix<uint32_t, -1, -1> F;
};

void get_position_and_color_from_vox(Vox &vox, PlyMesh &mesh, Eigen::Vector3f voxelsize) {
	std::vector<Eigen::Vector3f> positions;
	std::vector<Eigen::Matrix<uint8_t, 3, 1>> colors;
	int n_voxels = 0;
	for (int k = 0; k < vox.dims[2]; k++) {
		for (int j = 0; j < vox.dims[1]; j++) {
			for (int i = 0; i < vox.dims[0]; i++) {
				int index = k*vox.dims[1]*vox.dims[0] + j*vox.dims[0] + i;
				if (std::abs(vox.sdf[index]) <= inargs.trunc*vox.res) {
					Eigen::Vector3f p;
					p = (vox.grid2world*Eigen::Vector4f(i, j, k, 1)).topRows(3);
					positions.push_back(p);
					Eigen::Vector3f color;
					ColorMap::colormap(vox.pdf[index], color, inargs.cmap);
					colors.push_back((255.0f*color).cast<uint8_t>());
					//colors.push_back(Eigen::Vector3f(255, 0, 0));
					n_voxels++;
				}
			}	
		}	
	}
	
	int n_verts_per_voxel = box_vertices.cols();
	int n_elems_per_voxel = box_elements.cols();
	std::cout << "n_voxels: " << n_voxels << std::endl;

	mesh.V.resize(3, n_voxels*n_verts_per_voxel);
	mesh.C.resize(3, n_voxels*n_verts_per_voxel);
	mesh.N.resize(3, n_voxels*n_verts_per_voxel);
	mesh.F.resize(3, n_voxels*n_elems_per_voxel);

	Eigen::Vector3f res;
	res = 0.45*voxelsize*vox.res;

	for (int i = 0; i < n_voxels; i++) {
		Eigen::Vector3f p = positions[i];

		mesh.V.block(0, i*n_verts_per_voxel, 3, n_verts_per_voxel) = (box_vertices.array().colwise()*res.array()).colwise() + p.array();
		mesh.C.block(0, i*n_verts_per_voxel, 3, n_verts_per_voxel).colwise() = colors[i];
		mesh.N.block(0, i*n_verts_per_voxel, 3, n_verts_per_voxel) = box_normals;
		mesh.F.block(0, i*n_elems_per_voxel, 3, n_elems_per_voxel) = box_elements + Eigen::Matrix<uint32_t, -1, -1>::Constant(3, n_elems_per_voxel, i*n_verts_per_voxel);
	}
}

void write_ply(const std::string & filename, PlyMesh &mesh) {

	std::filebuf fb_binary;
	//fb_binary.open(filename, std::ios::out | std::ios::binary);
	fb_binary.open(filename, std::ios::out);
	std::ostream outstream_binary(&fb_binary);
	if (outstream_binary.fail()) throw std::runtime_error("failed to open " + filename);

	tinyply::PlyFile ply_file;
	typedef tinyply::Type Type;

	ply_file.add_properties_to_element("vertex", { "x", "y", "z" },  Type::FLOAT32, mesh.V.cols(), reinterpret_cast<uint8_t*>((float*)mesh.V.data()), Type::INVALID, 0);

	//ply_file.add_properties_to_element("vertex", { "nx", "ny", "nz" }, Type::FLOAT32, mesh.N.cols(), reinterpret_cast<uint8_t*>(mesh.N.data()), Type::INVALID, 0);
	
	ply_file.add_properties_to_element("vertex", { "red", "green", "blue" },  Type::UINT8, mesh.C.cols(), reinterpret_cast<uint8_t*>(mesh.C.data()), Type::INVALID, 0);

	ply_file.add_properties_to_element("face", { "vertex_indices" }, Type::UINT32, mesh.F.cols(), reinterpret_cast<uint8_t*>(mesh.F.data()), Type::UINT8, 3);

	ply_file.get_comments().push_back("generated by tinyply 2.2");

	// Write a binary file
	ply_file.write(outstream_binary, false);
}

void parse_args(int argc, char** argv) {
	args::ArgumentParser parser("This is a test program.", "This goes after the options.");
	args::Group allgroup(parser, "", args::Group::Validators::All);

	args::ValueFlag<std::string> in(allgroup, "bunny.vox", "vox file", {"in"});
	args::ValueFlag<std::string> out(allgroup, "bunny.ply", "out file", {"out"});
	args::ValueFlag<bool> is_unitless(parser, "false", "normalize voxel grid or no units?", {"is_unitless"}, false);
	args::ValueFlag<bool> redcenter(parser, "false", "red center in grid?", {"redcenter"}, false);
	args::ValueFlag<std::string> cmap(parser, "jet, inferno, magma, viridis, gray2red, beige2red", "color map format", {"cmap"}, "jet");
	args::ValueFlag<float> trunc(parser, "1.0", "truncation for visible voxels", {"trunc"}, 1.0);


	try {
		parser.ParseCLI(argc, argv);
	} catch (args::ParseError e) {
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		exit(1);
	} catch (args::ValidationError e) {
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		exit(1);
	}

	inargs.in = args::get(in);
	inargs.out = args::get(out);
	inargs.cmap = args::get(cmap);
	inargs.redcenter = args::get(redcenter);
	inargs.trunc = args::get(trunc);
	inargs.is_unitless = args::get(is_unitless);

};

int main(int argc, char** argv) {
   	parse_args(argc, argv); 
	Box3D::create(box_vertices, box_normals, box_elements);

	Vox vox;
	Eigen::Vector3f voxelsize(1, 1, 1);

	vox = load_vox(inargs.in);

	if (inargs.is_unitless) {
		Eigen::Vector3f t;
		Eigen::Quaternionf q;
		Eigen::Vector3f s;
		decompose_mat4(vox.grid2world, t, q, s);
		voxelsize = s;
	}

	if (vox.pdf.size() == 0) {
		vox.pdf.resize(vox.sdf.size());
		std::fill(vox.pdf.begin(), vox.pdf.end(), 0);
		if (inargs.redcenter) {
			int c = vox.dims(0)/2;
			int dim = vox.dims(0);
			int w = 1;
			for (int i = c - w; i < c + w + 1; i++)
				for (int j = c - w; j < c + w + 1; j++)
					for (int k = c - w; k < c + w + 1; k++)
						vox.pdf[i*dim*dim + j*dim + k] = 1;
		}
	}

	PlyMesh mesh;
	std::map<std::string, Eigen::MatrixXf*> hashmap;
	get_position_and_color_from_vox(vox, mesh, voxelsize);

	write_ply(inargs.out, mesh);



	return 0;
}
