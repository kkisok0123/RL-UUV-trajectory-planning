/*
 * dynamics_step.cpp
 *
 * Rewritten from a MATLAB MEX entry point into a CPython extension module.
 * The numerical core is unchanged: fixed-step RK4 integration plus the
 * hydrodynamic force model. Python usage after compilation:
 *
 *   import dynamics_step
 *   state_new, A1h, A2h, A3h, A4h, a5h = dynamics_step.update(
 *       fish_state, body_params, fin_params,
 *       dt, t_k,
 *       A1_ref, A2_ref, A3_ref, A4_ref, alpha5_ref,
 *       A1_hist, A2_hist, A3_hist, A4_hist, alpha5_hist,
 *       c_A, fin_f,
 *   )
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cmath>
#include <cstring>

/* ===== Constants ===== */
static const int NX = 13; /* State dimension */
static const int NY = 15; /* Integration grid points per axis */
static const int NX_GRID = 15;
static const double PI = 3.14159265358979323846;

/* ===== Body Parameters (indexed) ===== */
struct BodyParams {
  double m, G, B, rho_Fluid;
  double rBG_1, rBG_2, rBG_3;
  double J11, J12, J13, J21, J22, J23, J31, J32, J33;
  double A, L;
  double P1, P2_beta, P3_alpha, P2bar_alpha, P3bar_beta;
  double D2_w3, D3_w2, D1bar_w1, D2bar_w2, D3bar_w3;
  double lambda11, lambda22, lambda33, lambda44, lambda55, lambda66, lambda26,
      lambda35;
};

/* ===== Fin Parameters ===== */
struct FinParams {
  double rho_fluid, c_d;
  double LTOTAL, TF_LTOTAL, h;
  double pianyi_Y;
  /* c_right / d_right polynomial coefficients (relative to y - pianyi_Y) */
  /* c_right(y) = c0 + c1*dy + c2*dy^2 + ... + c6*dy^6  */
  double cr[7]; /* 7 coefficients for c_right */
  double dr[8]; /* 8 coefficients for d_right */
  /* tail fin boundary: e(y) = e0 + e2*y^2 + e4*y^4 + e6*y^6 */
  double e_coeff[4]; /* e0, e2, e4, e6 */
  /* Motor positions */
  double rBM_r[3], rBM_l[3], rBM_tf[3];
  double f; /* driving frequency */
};

/* ===== CPG amplitude function: A(t) = A_ref + delta_A * (1 + c/2 * t) *
 * exp(-c/2 * t) ===== */
static inline double cpg_A_val(double A_ref, double A_hist, double c_A,
                               double t) {
  double delta = A_hist - A_ref;
  double half_c = c_A / 2.0;
  return A_ref + delta * (1.0 + half_c * t) * exp(-half_c * t);
}

/* ===== CPG amplitude derivative ===== */
static inline double cpg_A_dot(double A_ref, double A_hist, double c_A,
                               double t) {
  double delta = A_hist - A_ref;
  double half_c = c_A / 2.0;
  /* d/dt [ delta * (1 + half_c*t) * exp(-half_c*t) ]
     = delta * [ half_c * exp(-half_c*t) - half_c*(1+half_c*t)*exp(-half_c*t) ]
     = delta * exp(-half_c*t) * [ half_c - half_c - half_c^2*t ]
     = -delta * half_c^2 * t * exp(-half_c*t) */
  return -delta * half_c * half_c * t * exp(-half_c * t);
}

/* ===== Angle functions ===== */
struct AngleState {
  double alpha1, alpha2, alpha3, alpha4, alpha5;
  double alpha1t, alpha2t, alpha3t, alpha4t, alpha5t;
};

static void compute_angles(double t_local, double t_k, double fin_f, double c_A,
                           double A1_ref, double A2_ref, double A3_ref,
                           double A4_ref, double alpha5_ref, double A1_hist,
                           double A2_hist, double A3_hist, double A4_hist,
                           double alpha5_hist, AngleState &as) {
  double A1 = cpg_A_val(A1_ref, A1_hist, c_A, t_local);
  double A2 = cpg_A_val(A2_ref, A2_hist, c_A, t_local);
  double A3 = cpg_A_val(A3_ref, A3_hist, c_A, t_local);
  double A4 = cpg_A_val(A4_ref, A4_hist, c_A, t_local);
  double A5 = cpg_A_val(alpha5_ref, alpha5_hist, c_A, t_local);

  double phase = 2.0 * PI * fin_f * (t_local + t_k);
  double sp = sin(phase);
  double cp = cos(phase);
  double omega = 2.0 * PI * fin_f;

  /* Angle values (from main_with_los.m lines 483-486) */
  as.alpha1 = A1 * sp;
  as.alpha2 = -A2 * cp;
  as.alpha3 = -A3 * sp;
  as.alpha4 = -A4 * cp;
  as.alpha5 = A5;

  /* Angle derivatives */
  double A1d = cpg_A_dot(A1_ref, A1_hist, c_A, t_local);
  double A2d = cpg_A_dot(A2_ref, A2_hist, c_A, t_local);
  double A3d = cpg_A_dot(A3_ref, A3_hist, c_A, t_local);
  double A4d = cpg_A_dot(A4_ref, A4_hist, c_A, t_local);
  double A5d = cpg_A_dot(alpha5_ref, alpha5_hist, c_A, t_local);

  as.alpha1t = A1d * sp + A1 * cp * omega;
  as.alpha2t = -A2d * cp + A2 * sp * omega;
  as.alpha3t = -A3d * sp - A3 * cp * omega;
  as.alpha4t = -A4d * cp + A4 * sp * omega;
  as.alpha5t = A5d;
}

/* ===== 3x3 matrix helpers ===== */
static void mat3_mul_vec3(const double M[9], const double v[3], double out[3]) {
  /* M stored row-major: M[row*3+col] */
  out[0] = M[0] * v[0] + M[1] * v[1] + M[2] * v[2];
  out[1] = M[3] * v[0] + M[4] * v[1] + M[5] * v[2];
  out[2] = M[6] * v[0] + M[7] * v[1] + M[8] * v[2];
}

static void mat3_transpose(const double M[9], double Mt[9]) {
  Mt[0] = M[0];
  Mt[1] = M[3];
  Mt[2] = M[6];
  Mt[3] = M[1];
  Mt[4] = M[4];
  Mt[5] = M[7];
  Mt[6] = M[2];
  Mt[7] = M[5];
  Mt[8] = M[8];
}

static void cross3(const double a[3], const double b[3], double c[3]) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

/* ===== Rotation matrices from define_fin_update.m ===== */
/* REe_r: body -> right pectoral fin */
static void make_REe_r(double a1, double a2, double M[9]) {
  double s1 = sin(a1), c1 = cos(a1);
  double s2 = sin(a2), c2 = cos(a2);
  M[0] = c2;
  M[1] = s1 * s2;
  M[2] = -c1 * s2;
  M[3] = 0.0;
  M[4] = c1;
  M[5] = s1;
  M[6] = s2;
  M[7] = -s1 * c2;
  M[8] = c1 * c2;
}

/* REe_l: body -> left pectoral fin */
static void make_REe_l(double a3, double a4, double M[9]) {
  double s3 = sin(a3), c3 = cos(a3);
  double s4 = sin(a4), c4 = cos(a4);
  M[0] = c4;
  M[1] = s3 * s4;
  M[2] = -c3 * s4;
  M[3] = 0.0;
  M[4] = c3;
  M[5] = s3;
  M[6] = s4;
  M[7] = -s3 * c4;
  M[8] = c3 * c4;
}

/* REe_tf: body -> tail fin */
static void make_REe_tf(double a5, double M[9]) {
  double s5 = sin(a5), c5 = cos(a5);
  M[0] = c5;
  M[1] = 0.0;
  M[2] = -s5;
  M[3] = 0.0;
  M[4] = 1.0;
  M[5] = 0.0;
  M[6] = s5;
  M[7] = 0.0;
  M[8] = c5;
}

/* ===== Fin boundary curves ===== */
static double eval_c_right(const FinParams &fp, double y) {
  double dy = y - fp.pianyi_Y;
  double dy2 = dy * dy, dy3 = dy2 * dy, dy4 = dy3 * dy, dy5 = dy4 * dy,
         dy6 = dy5 * dy;
  return fp.cr[0] + fp.cr[1] * dy + fp.cr[2] * dy2 + fp.cr[3] * dy3 +
         fp.cr[4] * dy4 + fp.cr[5] * dy5 + fp.cr[6] * dy6;
}

static double eval_d_right(const FinParams &fp, double y) {
  double dy = y - fp.pianyi_Y;
  double dy2 = dy * dy, dy3 = dy2 * dy, dy4 = dy3 * dy, dy5 = dy4 * dy,
         dy6 = dy5 * dy, dy7 = dy6 * dy;
  return fp.dr[0] + fp.dr[1] * dy + fp.dr[2] * dy2 + fp.dr[3] * dy3 +
         fp.dr[4] * dy4 + fp.dr[5] * dy5 + fp.dr[6] * dy6 + fp.dr[7] * dy7;
}

static double eval_c_left(const FinParams &fp, double y) {
  return eval_c_right(fp, -y);
}

static double eval_d_left(const FinParams &fp, double y) {
  return eval_d_right(fp, -y);
}

static double eval_tail_e(const FinParams &fp, double y) {
  double y2 = y * y, y4 = y2 * y2, y6 = y4 * y2;
  return fp.e_coeff[0] + fp.e_coeff[1] * y2 + fp.e_coeff[2] * y4 +
         fp.e_coeff[3] * y6;
}

/* ===== Fin force computation ===== */
/* Right pectoral fin force */
static void right_fin_force(const FinParams &fp, const AngleState &as,
                            const double v[3], const double w[3], double FB[3],
                            double MB[3]) {
  double REe[9], ReE[9];
  make_REe_r(as.alpha1, as.alpha2, REe);
  mat3_transpose(REe, ReE);

  /* Motor velocity in body frame */
  double wXr[3];
  cross3(w, fp.rBM_r, wXr);
  double vM_E[3] = {v[0] + wXr[0], v[1] + wXr[1], v[2] + wXr[2]};

  /* Transform to fin frame */
  double vM_e[3], wB_e[3];
  mat3_mul_vec3(REe, vM_E, vM_e);
  mat3_mul_vec3(REe, w, wB_e);

  /* Relative angular velocity of fin */
  double wM_e[3];
  wM_e[0] = as.alpha1t * cos(as.alpha2);
  wM_e[1] = as.alpha2t;
  wM_e[2] = as.alpha1t * sin(as.alpha2);

  /* Numerical integration over fin area */
  double y_min = fp.pianyi_Y;
  double y_max = fp.LTOTAL + fp.pianyi_Y;
  double dy = (y_max - y_min) / NY;

  double F3_e = 0.0, M1_e = 0.0, M2_e = 0.0;

  double wx_total = wB_e[0] + wM_e[0];
  double wy_total = wB_e[1] + wM_e[1];

  for (int i = 0; i < NY; i++) {
    double y = y_min + dy * (0.5 + i);
    double x_min_v = eval_d_right(fp, y);
    double x_max_v = eval_c_right(fp, y);
    if (x_max_v <= x_min_v)
      continue;

    double dx = (x_max_v - x_min_v) / NX_GRID;
    double dA = dx * dy;
    double base_vz = vM_e[2] + y * wx_total;

    double sum_F3 = 0.0, sum_M2_x = 0.0;
    for (int j = 0; j < NX_GRID; j++) {
      double x = x_min_v + dx * (0.5 + j);
      double Vrel = base_vz - x * wy_total;
      double P = -0.5 * fp.rho_fluid * fp.c_d * Vrel * fabs(Vrel);
      sum_F3 += P;
      sum_M2_x += P * x;
    }
    F3_e += sum_F3 * dA;
    M1_e += y * sum_F3 * dA;
    M2_e += -sum_M2_x * dA;
  }

  /* Transform back to body frame */
  double F_fin[3] = {0.0, 0.0, F3_e};
  double M_fin[3] = {M1_e, M2_e, 0.0};
  double FD_M_E[3], MD_M_E[3];
  mat3_mul_vec3(ReE, F_fin, FD_M_E);
  mat3_mul_vec3(ReE, M_fin, MD_M_E);

  /* Force at buoyancy center */
  FB[0] = FD_M_E[0];
  FB[1] = FD_M_E[1];
  FB[2] = FD_M_E[2];
  double rXF[3];
  cross3(fp.rBM_r, FB, rXF);
  MB[0] = MD_M_E[0] + rXF[0];
  MB[1] = MD_M_E[1] + rXF[1];
  MB[2] = MD_M_E[2] + rXF[2];
}

/* Left pectoral fin force */
static void left_fin_force(const FinParams &fp, const AngleState &as,
                           const double v[3], const double w[3], double FB[3],
                           double MB[3]) {
  double REe[9], ReE[9];
  make_REe_l(as.alpha3, as.alpha4, REe);
  mat3_transpose(REe, ReE);

  double wXr[3];
  cross3(w, fp.rBM_l, wXr);
  double vM_E[3] = {v[0] + wXr[0], v[1] + wXr[1], v[2] + wXr[2]};

  double vM_e[3], wB_e[3];
  mat3_mul_vec3(REe, vM_E, vM_e);
  mat3_mul_vec3(REe, w, wB_e);

  double wM_e[3];
  wM_e[0] = as.alpha3t * cos(as.alpha4);
  wM_e[1] = as.alpha4t;
  wM_e[2] = as.alpha3t * sin(as.alpha4);

  double y_min = -(fp.LTOTAL + fp.pianyi_Y);
  double y_max = -fp.pianyi_Y;
  double dy = (y_max - y_min) / NY;

  double F3_e = 0.0, M1_e = 0.0, M2_e = 0.0;
  double wx_total = wB_e[0] + wM_e[0];
  double wy_total = wB_e[1] + wM_e[1];

  for (int i = 0; i < NY; i++) {
    double y = y_min + dy * (0.5 + i);
    double x_min_v = eval_d_left(fp, y);
    double x_max_v = eval_c_left(fp, y);
    if (x_max_v <= x_min_v)
      continue;

    double dx = (x_max_v - x_min_v) / NX_GRID;
    double dA = dx * dy;
    double base_vz = vM_e[2] + y * wx_total;

    double sum_F3 = 0.0, sum_M2_x = 0.0;
    for (int j = 0; j < NX_GRID; j++) {
      double x = x_min_v + dx * (0.5 + j);
      double Vrel = base_vz - x * wy_total;
      double P = -0.5 * fp.rho_fluid * fp.c_d * Vrel * fabs(Vrel);
      sum_F3 += P;
      sum_M2_x += P * x;
    }
    F3_e += sum_F3 * dA;
    M1_e += y * sum_F3 * dA;
    M2_e += -sum_M2_x * dA;
  }

  double F_fin[3] = {0.0, 0.0, F3_e};
  double M_fin[3] = {M1_e, M2_e, 0.0};
  double FD_M_E[3], MD_M_E[3];
  mat3_mul_vec3(ReE, F_fin, FD_M_E);
  mat3_mul_vec3(ReE, M_fin, MD_M_E);

  FB[0] = FD_M_E[0];
  FB[1] = FD_M_E[1];
  FB[2] = FD_M_E[2];
  double rXF[3];
  cross3(fp.rBM_l, FB, rXF);
  MB[0] = MD_M_E[0] + rXF[0];
  MB[1] = MD_M_E[1] + rXF[1];
  MB[2] = MD_M_E[2] + rXF[2];
}

/* Tail fin force */
static void tail_fin_force(const FinParams &fp, const AngleState &as,
                           const double v[3], const double w[3], double FB[3],
                           double MB[3]) {
  double REe[9], ReE[9];
  make_REe_tf(as.alpha5, REe);
  mat3_transpose(REe, ReE);

  double wXr[3];
  cross3(w, fp.rBM_tf, wXr);
  double vM_E[3] = {v[0] + wXr[0], v[1] + wXr[1], v[2] + wXr[2]};

  double vM_e[3], wB_e[3];
  mat3_mul_vec3(REe, vM_E, vM_e);
  mat3_mul_vec3(REe, w, wB_e);

  double wM_e[3] = {0.0, as.alpha5t, 0.0};

  double y_min = -0.5 * fp.TF_LTOTAL;
  double y_max = 0.5 * fp.TF_LTOTAL;
  double dy = (y_max - y_min) / NY;

  double F3_e = 0.0, M2_e = 0.0;
  double wx_total = wB_e[0] + wM_e[0];
  double wy_total = wB_e[1] + wM_e[1];

  for (int i = 0; i < NY; i++) {
    double y = y_min + dy * (0.5 + i);
    double x_min_v = eval_tail_e(fp, y);
    double x_max_v = 0.0;
    if (x_max_v <= x_min_v)
      continue;

    double dx = (x_max_v - x_min_v) / NX_GRID;
    double dA = dx * dy;
    double base_vz = vM_e[2] + y * wx_total;

    double sum_F3 = 0.0, sum_M2_x = 0.0;
    for (int j = 0; j < NX_GRID; j++) {
      double x = x_min_v + dx * (0.5 + j);
      double Vrel = base_vz - x * wy_total;
      double P = -0.5 * fp.rho_fluid * fp.c_d * Vrel * fabs(Vrel);
      sum_F3 += P;
      sum_M2_x += P * x;
    }
    F3_e += sum_F3 * dA;
    M2_e += -sum_M2_x * dA;
  }

  double F_fin[3] = {0.0, 0.0, F3_e};
  double M_fin[3] = {0.0, M2_e, 0.0};
  double FD_M_E[3], MD_M_E[3];
  mat3_mul_vec3(ReE, F_fin, FD_M_E);
  mat3_mul_vec3(ReE, M_fin, MD_M_E);

  FB[0] = FD_M_E[0];
  FB[1] = FD_M_E[1];
  FB[2] = FD_M_E[2];
  double rXF[3];
  cross3(fp.rBM_tf, FB, rXF);
  MB[0] = MD_M_E[0] + rXF[0];
  MB[1] = MD_M_E[1] + rXF[1];
  MB[2] = MD_M_E[2] + rXF[2];
}

/* ===== 6x6 linear solve (Gaussian elimination with partial pivoting) ===== */
static void solve6x6(double A[6][6], double b[6], double x[6]) {
  /* Forward elimination with partial pivoting */
  for (int col = 0; col < 6; col++) {
    /* Find pivot */
    int maxRow = col;
    double maxVal = fabs(A[col][col]);
    for (int row = col + 1; row < 6; row++) {
      if (fabs(A[row][col]) > maxVal) {
        maxVal = fabs(A[row][col]);
        maxRow = row;
      }
    }
    /* Swap rows */
    if (maxRow != col) {
      for (int j = col; j < 6; j++) {
        double tmp = A[col][j];
        A[col][j] = A[maxRow][j];
        A[maxRow][j] = tmp;
      }
      double tmp = b[col];
      b[col] = b[maxRow];
      b[maxRow] = tmp;
    }
    /* Eliminate below */
    for (int row = col + 1; row < 6; row++) {
      double factor = A[row][col] / A[col][col];
      for (int j = col; j < 6; j++) {
        A[row][j] -= factor * A[col][j];
      }
      b[row] -= factor * b[col];
    }
  }
  /* Back substitution */
  for (int i = 5; i >= 0; i--) {
    x[i] = b[i];
    for (int j = i + 1; j < 6; j++) {
      x[i] -= A[i][j] * x[j];
    }
    x[i] /= A[i][i];
  }
}

/* ===== Main Dynamics Function ===== */
static void dynamics_rhs(const double x[NX], const BodyParams &bp,
                         const FinParams &fp, const AngleState &as,
                         double dxdt[NX]) {
  double vx = x[0], vy = x[1], vz = x[2];
  double wx = x[3], wy = x[4], wz = x[5];
  double q0 = x[6], q1 = x[7], q2 = x[8], q3 = x[9];

  /* Rotation matrix RIE (inertial to body) */
  double RIE[9];
  RIE[0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3;
  RIE[1] = 2 * (q1 * q2 + q0 * q3);
  RIE[2] = 2 * (q1 * q3 - q0 * q2);
  RIE[3] = 2 * (q1 * q2 - q0 * q3);
  RIE[4] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3;
  RIE[5] = 2 * (q0 * q1 + q3 * q2);
  RIE[6] = 2 * (q1 * q3 + q0 * q2);
  RIE[7] = 2 * (q2 * q3 - q0 * q1);
  RIE[8] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;

  /* Mass matrix M66 */
  double M[6][6];
  memset(M, 0, sizeof(M));
  M[0][0] = bp.m;
  M[0][4] = bp.m * bp.rBG_3;
  M[0][5] = -bp.m * bp.rBG_2;
  M[1][1] = bp.m;
  M[1][3] = -bp.m * bp.rBG_3;
  M[1][5] = bp.m * bp.rBG_1;
  M[2][2] = bp.m;
  M[2][3] = bp.m * bp.rBG_2;
  M[2][4] = -bp.m * bp.rBG_1;
  M[3][1] = -bp.m * bp.rBG_3;
  M[3][2] = bp.m * bp.rBG_2;
  M[3][3] = bp.J11;
  M[3][4] = -bp.J12;
  M[3][5] = -bp.J13;
  M[4][0] = bp.m * bp.rBG_3;
  M[4][2] = -bp.m * bp.rBG_1;
  M[4][3] = -bp.J21;
  M[4][4] = bp.J22;
  M[4][5] = -bp.J23;
  M[5][0] = -bp.m * bp.rBG_2;
  M[5][1] = bp.m * bp.rBG_1;
  M[5][3] = -bp.J31;
  M[5][4] = -bp.J32;
  M[5][5] = bp.J33;

  /* Added mass Lambda66 */
  double Lam[6][6];
  memset(Lam, 0, sizeof(Lam));
  Lam[0][0] = bp.lambda11;
  Lam[1][1] = bp.lambda22;
  Lam[1][5] = bp.lambda26;
  Lam[2][2] = bp.lambda33;
  Lam[2][4] = bp.lambda35;
  Lam[3][3] = bp.lambda44;
  Lam[4][2] = bp.lambda35;
  Lam[4][4] = bp.lambda55;
  Lam[5][1] = bp.lambda26;
  Lam[5][5] = bp.lambda66;

  /* (M + Lambda) */
  double ML[6][6];
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 6; j++)
      ML[i][j] = M[i][j] + Lam[i][j];

  /* Omega66 + L66 */
  double OL[6][6];
  memset(OL, 0, sizeof(OL));
  /* Omega66 */
  OL[0][1] = -wz;
  OL[0][2] = wy;
  OL[1][0] = wz;
  OL[1][2] = -wx;
  OL[2][0] = -wy;
  OL[2][1] = wx;
  OL[3][4] = -wz;
  OL[3][5] = wy;
  OL[4][3] = wz;
  OL[4][5] = -wx;
  OL[5][3] = -wy;
  OL[5][4] = wx;
  /* L66 additions */
  OL[3][1] += -vz;
  OL[3][2] += vy;
  OL[4][0] += vz;
  OL[4][2] += -vx;
  OL[5][0] += -vy;
  OL[5][1] += vx;

  /* vel6 = [vx vy vz wx wy wz] */
  double vel6[6] = {vx, vy, vz, wx, wy, wz};

  /* (M+Lam) * vel6 */
  double MLv[6];
  for (int i = 0; i < 6; i++) {
    MLv[i] = 0;
    for (int j = 0; j < 6; j++)
      MLv[i] += ML[i][j] * vel6[j];
  }

  /* -(OL) * MLv */
  double neg_OL_MLv[6];
  for (int i = 0; i < 6; i++) {
    neg_OL_MLv[i] = 0;
    for (int j = 0; j < 6; j++)
      neg_OL_MLv[i] -= OL[i][j] * MLv[j];
  }

  /* Buoyancy force */
  double grav_world[3] = {0, 0, -bp.B};
  double fw[3];
  mat3_mul_vec3(RIE, grav_world, fw);

  /* Gravity force */
  double grav_world2[3] = {0, 0, bp.G};
  double fg[3];
  mat3_mul_vec3(RIE, grav_world2, fg);
  /* Gravity moment: cross(rBG, fg) */
  double rBG[3] = {bp.rBG_1, bp.rBG_2, bp.rBG_3};
  double mg[3];
  cross3(rBG, fg, mg);

  /* Position forces */
  double alpha_a = atan2(vz, vx);
  double beta_a = atan2(vy, sqrt(vx * vx + vz * vz));
  double pd = 0.5 * bp.rho_Fluid * (vx * vx + vy * vy + vz * vz);

  double Fp[6];
  Fp[0] = bp.A * pd * bp.P1;
  Fp[1] = bp.A * pd * (bp.P2_beta * beta_a + bp.P1 * sin(beta_a));
  Fp[2] = bp.A * pd * (bp.P3_alpha * alpha_a + bp.P1 * sin(alpha_a));
  Fp[3] = 0;
  Fp[4] = bp.A * bp.L * pd * bp.P2bar_alpha * alpha_a;
  Fp[5] = bp.A * bp.L * pd * bp.P3bar_beta * beta_a;

  /* Damping forces */
  double qdyn = 0.5 * bp.rho_Fluid * sqrt(vx * vx + vy * vy + vz * vz) * bp.A;
  double Fd[6];
  Fd[0] = 0;
  Fd[1] = qdyn * bp.L * bp.D2_w3 * wz;
  Fd[2] = qdyn * bp.L * bp.D3_w2 * wy;
  Fd[3] = qdyn * bp.L * bp.L * bp.D1bar_w1 * wx;
  Fd[4] = qdyn * bp.L * bp.L * bp.D2bar_w2 * wy;
  Fd[5] = qdyn * bp.L * bp.L * bp.D3bar_w3 * wz;

  /* Fin forces */
  double v3[3] = {vx, vy, vz};
  double w3[3] = {wx, wy, wz};

  double FB_r[3], MB_r[3], FB_l[3], MB_l[3], FB_t[3], MB_t[3];
  right_fin_force(fp, as, v3, w3, FB_r, MB_r);
  left_fin_force(fp, as, v3, w3, FB_l, MB_l);
  tail_fin_force(fp, as, v3, w3, FB_t, MB_t);

  /* Total force vector F61 */
  double F61[6];
  F61[0] = fw[0] + fg[0] + Fp[0] + Fd[0] + FB_r[0] + FB_l[0] + FB_t[0];
  F61[1] = fw[1] + fg[1] + Fp[1] + Fd[1] + FB_r[1] + FB_l[1] + FB_t[1];
  F61[2] = fw[2] + fg[2] + Fp[2] + Fd[2] + FB_r[2] + FB_l[2] + FB_t[2];
  F61[3] = 0 + mg[0] + Fp[3] + Fd[3] + MB_r[0] + MB_l[0] + MB_t[0];
  F61[4] = 0 + mg[1] + Fp[4] + Fd[4] + MB_r[1] + MB_l[1] + MB_t[1];
  F61[5] = 0 + mg[2] + Fp[5] + Fd[5] + MB_r[2] + MB_l[2] + MB_t[2];

  /* RHS: (M+Lambda) \ (-OL*(M+Lambda)*vel6 + F61) */
  double rhs[6];
  for (int i = 0; i < 6; i++)
    rhs[i] = neg_OL_MLv[i] + F61[i];

  /* Solve ML * dPdt = rhs */
  /* Need to copy ML because solve modifies it */
  double ML_copy[6][6];
  memcpy(ML_copy, ML, sizeof(ML));
  double dPdt[6];
  solve6x6(ML_copy, rhs, dPdt);

  /* Quaternion kinematics */
  double dQdt[4];
  dQdt[0] = 0.5 * (-(q1 * wx + q2 * wy + q3 * wz));
  dQdt[1] = 0.5 * (q0 * wx + q2 * wz - q3 * wy);
  dQdt[2] = 0.5 * (q0 * wy - q1 * wz + q3 * wx);
  dQdt[3] = 0.5 * (q0 * wz + q1 * wy - q2 * wx);

  /* Position kinematics: dPos/dt = RIE^T * v */
  double RIE_T[9];
  mat3_transpose(RIE, RIE_T);
  double v_body[3] = {vx, vy, vz};
  double dPos[3];
  mat3_mul_vec3(RIE_T, v_body, dPos);

  /* Pack output */
  for (int i = 0; i < 6; i++)
    dxdt[i] = dPdt[i];
  for (int i = 0; i < 4; i++)
    dxdt[6 + i] = dQdt[i];
  for (int i = 0; i < 3; i++)
    dxdt[10 + i] = dPos[i];
}

/* ===== Fixed-step RK4 Integrator ===== */
static void rk4_step(double x[NX], double h, const BodyParams &bp,
                     const FinParams &fp, double t_local, double t_k,
                     double c_A, double fin_f, double A1_ref, double A2_ref,
                     double A3_ref, double A4_ref, double alpha5_ref,
                     double A1_hist, double A2_hist, double A3_hist,
                     double A4_hist, double alpha5_hist) {
  double k1[NX], k2[NX], k3[NX], k4[NX], xtmp[NX];
  AngleState as;

  /* k1 */
  compute_angles(t_local, t_k, fin_f, c_A, A1_ref, A2_ref, A3_ref, A4_ref,
                 alpha5_ref, A1_hist, A2_hist, A3_hist, A4_hist, alpha5_hist,
                 as);
  dynamics_rhs(x, bp, fp, as, k1);

  /* k2 */
  for (int i = 0; i < NX; i++)
    xtmp[i] = x[i] + 0.5 * h * k1[i];
  compute_angles(t_local + 0.5 * h, t_k, fin_f, c_A, A1_ref, A2_ref, A3_ref,
                 A4_ref, alpha5_ref, A1_hist, A2_hist, A3_hist, A4_hist,
                 alpha5_hist, as);
  dynamics_rhs(xtmp, bp, fp, as, k2);

  /* k3 */
  for (int i = 0; i < NX; i++)
    xtmp[i] = x[i] + 0.5 * h * k2[i];
  compute_angles(t_local + 0.5 * h, t_k, fin_f, c_A, A1_ref, A2_ref, A3_ref,
                 A4_ref, alpha5_ref, A1_hist, A2_hist, A3_hist, A4_hist,
                 alpha5_hist, as);
  dynamics_rhs(xtmp, bp, fp, as, k3);

  /* k4 */
  for (int i = 0; i < NX; i++)
    xtmp[i] = x[i] + h * k3[i];
  compute_angles(t_local + h, t_k, fin_f, c_A, A1_ref, A2_ref, A3_ref, A4_ref,
                 alpha5_ref, A1_hist, A2_hist, A3_hist, A4_hist, alpha5_hist,
                 as);
  dynamics_rhs(xtmp, bp, fp, as, k4);

  /* Update */
  for (int i = 0; i < NX; i++)
    x[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
}

static void load_body_params_from_array(const double bp_arr[36], BodyParams &bp) {
  int bi = 0;
  bp.m = bp_arr[bi++];
  bp.G = bp_arr[bi++];
  bp.B = bp_arr[bi++];
  bp.rho_Fluid = bp_arr[bi++];
  bp.rBG_1 = bp_arr[bi++];
  bp.rBG_2 = bp_arr[bi++];
  bp.rBG_3 = bp_arr[bi++];
  bp.J11 = bp_arr[bi++];
  bp.J12 = bp_arr[bi++];
  bp.J13 = bp_arr[bi++];
  bp.J21 = bp_arr[bi++];
  bp.J22 = bp_arr[bi++];
  bp.J23 = bp_arr[bi++];
  bp.J31 = bp_arr[bi++];
  bp.J32 = bp_arr[bi++];
  bp.J33 = bp_arr[bi++];
  bp.A = bp_arr[bi++];
  bp.L = bp_arr[bi++];
  bp.P1 = bp_arr[bi++];
  bp.P2_beta = bp_arr[bi++];
  bp.P3_alpha = bp_arr[bi++];
  bp.P2bar_alpha = bp_arr[bi++];
  bp.P3bar_beta = bp_arr[bi++];
  bp.D2_w3 = bp_arr[bi++];
  bp.D3_w2 = bp_arr[bi++];
  bp.D1bar_w1 = bp_arr[bi++];
  bp.D2bar_w2 = bp_arr[bi++];
  bp.D3bar_w3 = bp_arr[bi++];
  bp.lambda11 = bp_arr[bi++];
  bp.lambda22 = bp_arr[bi++];
  bp.lambda33 = bp_arr[bi++];
  bp.lambda44 = bp_arr[bi++];
  bp.lambda55 = bp_arr[bi++];
  bp.lambda66 = bp_arr[bi++];
  bp.lambda26 = bp_arr[bi++];
  bp.lambda35 = bp_arr[bi++];
}

static void load_fin_params_from_array(const double fp_arr[38], FinParams &fp) {
  int fi = 0;
  fp.rho_fluid = fp_arr[fi++];
  fp.c_d = fp_arr[fi++];
  fp.LTOTAL = fp_arr[fi++];
  fp.TF_LTOTAL = fp_arr[fi++];
  fp.h = fp_arr[fi++];
  fp.pianyi_Y = fp_arr[fi++];
  for (int j = 0; j < 7; j++)
    fp.cr[j] = fp_arr[fi++];
  for (int j = 0; j < 8; j++)
    fp.dr[j] = fp_arr[fi++];
  for (int j = 0; j < 4; j++)
    fp.e_coeff[j] = fp_arr[fi++];
  fp.rBM_r[0] = fp_arr[fi++];
  fp.rBM_r[1] = fp_arr[fi++];
  fp.rBM_r[2] = fp_arr[fi++];
  fp.rBM_l[0] = fp_arr[fi++];
  fp.rBM_l[1] = fp_arr[fi++];
  fp.rBM_l[2] = fp_arr[fi++];
  fp.rBM_tf[0] = fp_arr[fi++];
  fp.rBM_tf[1] = fp_arr[fi++];
  fp.rBM_tf[2] = fp_arr[fi++];
  fp.f = fp_arr[fi++];
}

static void simulate_step_arrays(double state[NX], const double bp_arr[36],
                                 const double fp_arr[38], double dt_total,
                                 double t_k, const double refs[5],
                                 const double hist_in[5], double c_A,
                                 double fin_f, double hist_out[5]) {
  BodyParams bp;
  FinParams fp;
  load_body_params_from_array(bp_arr, bp);
  load_fin_params_from_array(fp_arr, fp);

  double h_max = 1.0 / (40.0 * fin_f);
  int n_steps = (int)ceil(dt_total / h_max);
  if (n_steps < 1)
    n_steps = 1;
  double h = dt_total / n_steps;

  double t_local = 0.0;
  for (int step = 0; step < n_steps; step++) {
    rk4_step(state, h, bp, fp, t_local, t_k, c_A, fin_f, refs[0], refs[1],
             refs[2], refs[3], refs[4], hist_in[0], hist_in[1], hist_in[2],
             hist_in[3], hist_in[4]);
    t_local += h;
  }

  double qnorm = sqrt(state[6] * state[6] + state[7] * state[7] +
                      state[8] * state[8] + state[9] * state[9]);
  if (qnorm > 1e-12) {
    state[6] /= qnorm;
    state[7] /= qnorm;
    state[8] /= qnorm;
    state[9] /= qnorm;
  }

  hist_out[0] = cpg_A_val(refs[0], hist_in[0], c_A, dt_total);
  hist_out[1] = cpg_A_val(refs[1], hist_in[1], c_A, dt_total);
  hist_out[2] = cpg_A_val(refs[2], hist_in[2], c_A, dt_total);
  hist_out[3] = cpg_A_val(refs[3], hist_in[3], c_A, dt_total);
  hist_out[4] = cpg_A_val(refs[4], hist_in[4], c_A, dt_total);
}

#ifndef FISH_DYNAMICS_PYBIND
static bool load_double_sequence(PyObject *obj, int expected_len,
                                 double *out, const char *name) {
  PyObject *seq = PySequence_Fast(obj, name);
  if (seq == nullptr)
    return false;

  Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
  if (n != expected_len) {
    PyErr_Format(PyExc_ValueError, "%s must have length %d (got %zd)", name,
                 expected_len, n);
    Py_DECREF(seq);
    return false;
  }

  PyObject **items = PySequence_Fast_ITEMS(seq);
  for (int i = 0; i < expected_len; ++i) {
    double v = PyFloat_AsDouble(items[i]);
    if (PyErr_Occurred()) {
      PyErr_Format(PyExc_TypeError, "%s must contain only real numbers", name);
      Py_DECREF(seq);
      return false;
    }
    out[i] = v;
  }

  Py_DECREF(seq);
  return true;
}

static PyObject *build_state_list(const double state[NX]) {
  PyObject *list = PyList_New(NX);
  if (list == nullptr)
    return nullptr;

  for (int i = 0; i < NX; ++i) {
    PyObject *value = PyFloat_FromDouble(state[i]);
    if (value == nullptr) {
      Py_DECREF(list);
      return nullptr;
    }
    PyList_SET_ITEM(list, i, value);
  }
  return list;
}

static PyObject *py_update(PyObject *self, PyObject *args) {
  (void)self;

  PyObject *fish_state_obj = nullptr;
  PyObject *body_params_obj = nullptr;
  PyObject *fin_params_obj = nullptr;

  double dt_total, t_k;
  double A1_ref, A2_ref, A3_ref, A4_ref, alpha5_ref;
  double A1_hist, A2_hist, A3_hist, A4_hist, alpha5_hist;
  double c_A, fin_f;

  if (!PyArg_ParseTuple(
          args, "OOOdddddddddddddd",
          &fish_state_obj, &body_params_obj, &fin_params_obj,
          &dt_total, &t_k,
          &A1_ref, &A2_ref, &A3_ref, &A4_ref, &alpha5_ref,
          &A1_hist, &A2_hist, &A3_hist, &A4_hist, &alpha5_hist,
          &c_A, &fin_f)) {
    return nullptr;
  }

  double state[NX];
  if (!load_double_sequence(fish_state_obj, NX, state, "fish_state"))
    return nullptr;

  double bp_arr[36];
  if (!load_double_sequence(body_params_obj, 36, bp_arr, "body_params"))
    return nullptr;

  double fp_arr[38];
  if (!load_double_sequence(fin_params_obj, 38, fp_arr, "fin_params"))
    return nullptr;

  if (fin_f <= 0.0) {
    PyErr_SetString(PyExc_ValueError, "fin_f must be positive");
    return nullptr;
  }
  if (dt_total < 0.0) {
    PyErr_SetString(PyExc_ValueError, "dt must be non-negative");
    return nullptr;
  }

  const double refs[5] = {A1_ref, A2_ref, A3_ref, A4_ref, alpha5_ref};
  const double hist_in[5] = {A1_hist, A2_hist, A3_hist, A4_hist, alpha5_hist};
  double hist_out[5];
  simulate_step_arrays(state, bp_arr, fp_arr, dt_total, t_k, refs, hist_in,
                       c_A, fin_f, hist_out);

  PyObject *state_out = build_state_list(state);
  if (state_out == nullptr)
    return nullptr;

  PyObject *result =
      Py_BuildValue("(Oddddd)", state_out, hist_out[0], hist_out[1], hist_out[2],
                    hist_out[3], hist_out[4]);
  Py_DECREF(state_out);
  return result;
}

static PyMethodDef module_methods[] = {
    {"update", py_update, METH_VARARGS,
     "Advance the 13-state fish dynamics model by one timestep."},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "dynamics_step",
    "Python binding for the RK4 fish dynamics step.",
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit_dynamics_step(void) {
  return PyModule_Create(&module_def);
}
#endif
