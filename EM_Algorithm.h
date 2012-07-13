# void ReadCSV(vector<string> &record, const string& line, char delimiter);
# void kmeans(int dim, double *X, int n, int k, double *cluster_centroid, int *cluster_assignment_final);
# vector<double> * ParseCSV(int argc, char *argv[]);
# void fail(char *str);
# double euclid_distance(int dim, double *pl, double *p2);
# void all_distances(int dim, int n, int k, double *X, double *centroid, double *distance_out);
# double total_distance(int dim, int n, int k, double *X, double *centroids, int *cluster_assignment_index);
# double calc_total_distance(int dim, int n, int k, double *X, double *centroids, int *cluster_assignment_index);
# void choose_all_clusters_from_distances(int dim, int n, int k, double *X, double *distance_array, int *cluster_assignment_index);
# void calc_cluster_centroids(int dim, int n, int k, double *X, int *cluster_assignment_index, double *new_cluster_centroid);
# void get_cluster_member_count(int n, int k, int *cluster_assignment_index, int *cluster_member_count);
# void update_delta_score_table(int dim, int n, int k, double *X, int *cluster_assignment_cur, double *cluster_centroid, int *cluster_member_count,       double*point_move_score_table, int cc);
# void perform_move (int dim, int n, int k, double *X, int *cluster_assignment, double *cluster_centroid, int *cluster_member_count, int move_point, int move_target_cluster);
# void cluster_diag(int dim, int n, int k, double *X, int *cluster_assignment_index, double *cluster_centroid);
# void copy_assignment_array(int n, int *src, int *tgt);
# int assignment_change_count (int n, int a[], int b[]);



