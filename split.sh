for i in {001..720};do
cdo seltimestep,$i ROMS_1x48_output_southern_BoB_0.5x0.5degree.nc ROMS_1x48_output_southern_BoB_0.5x0.5degree_$i.nc
done
