#include "myCeres.h"
struct ReprojectCost
{
	cv::Point2d observation;

	ReprojectCost(cv::Point2d& observation)
		: observation(observation)
	{
	}

	template <typename T>
	bool operator()(const T* const intrinsic, const T* const extrinsic, const T* const pos3d, T* residuals) const
	{
		const T* r = extrinsic;
		const T* t = &extrinsic[3];

		T pos_proj[3];
		ceres::AngleAxisRotatePoint(r, pos3d, pos_proj);

		// Apply the camera translation
		pos_proj[0] += t[0];
		pos_proj[1] += t[1];
		pos_proj[2] += t[2];

		const T x = pos_proj[0] / pos_proj[2];
		const T y = pos_proj[1] / pos_proj[2];

		const T fx = intrinsic[0];
		const T fy = intrinsic[1];
		const T cx = intrinsic[2];
		const T cy = intrinsic[3];

		// Apply intrinsic
		const T u = fx * x + cx;// 0.01066666667
		const T v = fy * y + cy;// 0.01 

		residuals[0] = u - T(observation.x);
		residuals[1] = v - T(observation.y);

		return true;
	}
};

vector<double> Bundle_Adjustment(
	cv::Mat& intrinsic,
	vector<cv::Mat> extrinsics,
	vector< vector<int> >& correspond_struct_idx,
	vector< vector<cv::Point2f> >& key_points_for_all,
	vector<cv::Point3d>& structure,
	bool consFlag
	)
{
	ceres::Problem problem;

	// load extrinsics (rotations and motions)
	for (size_t i = 0; i < extrinsics.size(); ++i)
	{
		problem.AddParameterBlock(extrinsics[i].ptr<double>(), 6);
	}
	// fix the first camera.
	problem.SetParameterBlockConstant(extrinsics[0].ptr<double>());

	// load intrinsic
	problem.AddParameterBlock(intrinsic.ptr<double>(), 4); // fx, fy, cx, cy
	if(consFlag)
	  problem.SetParameterBlockConstant(intrinsic.ptr<double>());
	
	// load points
	ceres::LossFunction* loss_function;
	if(huberFlag)
	  loss_function = new ceres::HuberLoss(4);   // loss function make bundle adjustment robuster HuberLoss SoftLOneLoss
	else
	  loss_function = new ceres::SoftLOneLoss(4);
	for (size_t img_idx = 0; img_idx < correspond_struct_idx.size(); ++img_idx)
	{
		vector<int>& point3d_ids = correspond_struct_idx[img_idx];
		vector<cv::Point2f>& key_points = key_points_for_all[img_idx];
		for (size_t point_idx = 0; point_idx < point3d_ids.size(); ++point_idx)
		{
			int point3d_id = point3d_ids[point_idx];
			if (point3d_id < 0)
				continue;

			cv::Point2d observed = key_points[point_idx];
			// 模板参数中，第一个为代价函数的类型，第二个为代价的维度，剩下三个分别为代价函数第一第二还有第三个参数的维度
			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(new ReprojectCost(observed));

			problem.AddResidualBlock(
				cost_function,
				loss_function,
				intrinsic.ptr<double>(),            // Intrinsic
				extrinsics[img_idx].ptr<double>(),  // View Rotation and Translation
				&(structure[point3d_id].x)          // Point in 3D space
				);
		}
	}

	// Solve BA
	ceres::Solver::Options ceres_config_options;
	ceres_config_options.minimizer_progress_to_stdout = true;
	ceres_config_options.logging_type = ceres::SILENT;
	ceres_config_options.num_threads = 1;
	ceres_config_options.preconditioner_type = ceres::JACOBI;
	ceres_config_options.linear_solver_type = ceres::DENSE_SCHUR;//SPARSE_NORMAL_CHOLESKY;;SPARSE_SCHUR
	ceres_config_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;

	ceres::Solver::Summary summary;
	ceres::Solve(ceres_config_options, &problem, &summary);
	
	vector<double> blockError(2);
	if (!summary.IsSolutionUsable())
	{
		std::cout << "Bundle Adjustment failed." << std::endl;
		blockError[0] = 10.0;
		blockError[1] = 10.0;
	}
	else
	{
		// Display statistics about the minimization
		std::cout << std::endl
			<< "Bundle Adjustment statistics (approximated RMSE):\n"
			<< " #views: " << extrinsics.size() << "\n"
			<< " #residuals: " << summary.num_residuals << "\n"
			<< " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
			<< " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
			<< " Time (s): " << summary.total_time_in_seconds << "\n"
			<< std::endl;
		 blockError[0] = std::sqrt(summary.initial_cost / summary.num_residuals);
		 blockError[1] = std::sqrt(summary.final_cost / summary.num_residuals);
	}
	return blockError;
}
