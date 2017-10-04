#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

int main()
{

        /* runtime parameters */
        double ox, oy, oz;
        double dx, dy, dz;
        int nx, ny, nz;

        int flag;
        int numpars;
        int dir;

        /* input and output files */
        char sampout[20];
        FILE *fp;

        //local variables
        int i, j, k;
        double xx, yy, zz;
        double r1, r2, r3, r4; 


        printf("Enter name of output file: \n");
        scanf("%s", sampout);
        //sampout = "out.txt";

        printf("Grid (0), uniform rand (1), normal rand (2), grid bdry (3): \n");
        scanf("%d", &flag);

        if (flag == 0 || flag == 3) {
            printf("Enter origin x, y, z: \n");
            scanf("%lf %lf %lf", &ox, &oy, &oz);

            printf("Enter grid spacing x, y, z: \n");
            scanf("%lf %lf %lf", &dx, &dy, &dz);

            printf("Enter number gridpoints nx, ny, nz: \n");
            scanf("%d %d %d", &nx, &ny, &nz);

            printf("Enter primary direction (0--x, 1--y, 2--z): \n");
            scanf("%d", &dir);

            /* Writing coordinates for the targets */
            fp = fopen(sampout, "wb");
            if (!fp) {
                perror("File opening failed");
                return EXIT_FAILURE;
            }
            if (flag == 0) {
                if (dir == 0) {
                    for (i = 0; i < nx; i++) {
                        for (j = 0; j < ny; j++) {
                            for (k = 0; k < nz; k++) {
                                xx = ox + i*dx;
                                yy = oy + j*dy;
                                zz = oz + k*dz;
                                fwrite(&xx, sizeof(double), 1, fp);
                                fwrite(&yy, sizeof(double), 1, fp);
                                fwrite(&zz, sizeof(double), 1, fp);
                            }
                        }
                    }
                } else if (dir == 1) {
                    for (j = 0; j < ny; j++) {
                        for (k = 0; k < nz; k++) {
                            for (i = 0; i < nx; i++) {
                                xx = ox + i*dx;
                                yy = oy + j*dy;
                                zz = oz + k*dz;
                                fwrite(&xx, sizeof(double), 1, fp);
                                fwrite(&yy, sizeof(double), 1, fp);
                                fwrite(&zz, sizeof(double), 1, fp);
                            }
                        }
                    }
                } else if (dir == 2) {
                    for (k = 0; k < nz; k++) {
                        for (i = 0; i < nx; i++) {
                            for (j = 0; j < ny; j++) {
                                xx = ox + i*dx;
                                yy = oy + j*dy;
                                zz = oz + k*dz;
                                fwrite(&xx, sizeof(double), 1, fp);
                                fwrite(&yy, sizeof(double), 1, fp);
                                fwrite(&zz, sizeof(double), 1, fp);
                            }
                        }
                    }
                }
                
            } else if (flag == 3) {
                
                zz = oz;
                for (i = 0; i < nx; i++) {
                    for (j = 0; j < ny; j++) {
                        xx = ox + i*dx;
                        yy = oy + j*dy;
                        fwrite(&xx, sizeof(double), 1, fp);
                        fwrite(&yy, sizeof(double), 1, fp);
                        fwrite(&zz, sizeof(double), 1, fp);
                    }
                }
                
                zz = oz + (nz-1)*dz;
                for (i = 0; i < nx; i++) {
                    for (j = 0; j < ny; j++) {
                        xx = ox + i*dx;
                        yy = oy + j*dy;
                        fwrite(&xx, sizeof(double), 1, fp);
                        fwrite(&yy, sizeof(double), 1, fp);
                        fwrite(&zz, sizeof(double), 1, fp);
                    }
                }
                
                yy = oy;
                for (i = 0; i < nx; i++) {
                    for (k = 1; k < nz-1; k++) {
                        xx = ox + i*dx;
                        zz = oz + k*dz;
                        fwrite(&xx, sizeof(double), 1, fp);
                        fwrite(&yy, sizeof(double), 1, fp);
                        fwrite(&zz, sizeof(double), 1, fp);
                    }
                }
                
                yy = oy + (ny-1)*dy;
                for (i = 0; i < nx; i++) {
                    for (k = 1; k < nz-1; k++) {
                        xx = ox + i*dx;
                        zz = oz + k*dz;
                        fwrite(&xx, sizeof(double), 1, fp);
                        fwrite(&yy, sizeof(double), 1, fp);
                        fwrite(&zz, sizeof(double), 1, fp);
                    }
                }
                
                xx = ox;
                for (j = 1; j < ny-1; j++) {
                    for (k = 1; k < nz-1; k++) {
                        yy = oy + j*dy;
                        zz = oz + k*dz;
                        fwrite(&xx, sizeof(double), 1, fp);
                        fwrite(&yy, sizeof(double), 1, fp);
                        fwrite(&zz, sizeof(double), 1, fp);
                    }
                }
                
                xx = ox + (nx-1)*dx;
                for (j = 1; j < ny-1; j++) {
                    for (k = 1; k < nz-1; k++) {
                        yy = oy + j*dy;
                        zz = oz + k*dz;
                        fwrite(&xx, sizeof(double), 1, fp);
                        fwrite(&yy, sizeof(double), 1, fp);
                        fwrite(&zz, sizeof(double), 1, fp);
                    }
                }
                
            }

            fclose(fp);

        } else if (flag == 1 || flag == 2) {

            printf("Enter number of particles: \n");
            scanf("%d", &numpars);

            printf("Enter left limits x, y, z: \n");
            scanf("%lf %lf %lf", &ox, &oy, &oz);

            printf("Enter right limits x, y, z: \n");
            scanf("%lf %lf %lf", &dx, &dy, &dz);

            /* Writing coordinates for the targets */
            fp = fopen(sampout, "wb");
            if (!fp) {
                perror("File opening failed");
                return EXIT_FAILURE;
            }

            srand(time(NULL));

            if (flag == 1) {
                for (i = 0; i < numpars; i++) {
                    xx = ox + (dx-ox) * ((double)rand())/RAND_MAX;
                    yy = oy + (dy-oy) * ((double)rand())/RAND_MAX;
                    zz = oz + (dz-oz) * ((double)rand())/RAND_MAX;

                    fwrite(&xx, sizeof(double), 1, fp);
                    fwrite(&yy, sizeof(double), 1, fp);
                    fwrite(&zz, sizeof(double), 1, fp);
                }

            } else if (flag == 1 || flag == 2) {
                for (i = 0; i < numpars; i++) {
                    r1 = ((double)rand())/RAND_MAX;
                    r2 = ((double)rand())/RAND_MAX;
                    r3 = ((double)rand())/RAND_MAX;
                    r4 = ((double)rand())/RAND_MAX;

                    xx = sqrt(-2.0 * log(r1)) * cos(2.0 * M_PI * r2);
                    yy = sqrt(-2.0 * log(r1)) * sin(2.0 * M_PI * r2);
                    zz = sqrt(-2.0 * log(r3)) * cos(2.0 * M_PI * r4);

                    xx = sqrt(dx-ox) * xx + (dx+ox) / 2.0;
                    yy = sqrt(dy-oy) * yy + (dy+oy) / 2.0;
                    xx = sqrt(dz-oz) * zz + (dz+oz) / 2.0;

                    fwrite(&xx, sizeof(double), 1, fp);
                    fwrite(&yy, sizeof(double), 1, fp);
                    fwrite(&zz, sizeof(double), 1, fp);
                }
            }

            fclose(fp);

        }

        return 0;

}
