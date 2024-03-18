file = '/home/bbyrne/GEOSFP.20110101.CN.4x5.nc'
fileID = NCDF_Open(file, /NOWRITE)
vID = NCDF_VARID(fileid, 'lat')
NCDF_VARGET, fileId, vId, GC_lat
vID = NCDF_VARID(fileid, 'lon')
NCDF_VARGET, fileId, vId, GC_lon
vID = NCDF_VARID(fileid, 'FRLAND') ; Fraction of land in grid box 
NCDF_VARGET, fileId, vId, FRLAND


MTYPE = CTM_TYPE( 'GENERIC', RESOLUTION=[0.25,0.25])
GRIDINFO = CTM_GRID( MTYPE ) 
XMID = GRIDINFO.XMID   
YMID = GRIDINFO.YMID   
;PMID = GRIDINFO.PMID   
;ZMID = GRIDINFO.ZMID   
;XEDGE = GRIDINFO.XEDGE 
;YEDGE = GRIDINFO.YEDGE
   
grid_area = CTM_BOXSIZE( GRIDINFO, /M2 )

MTYPE2 = CTM_TYPE( 'GEOS5', RESOLUTION=2 )
GRIDINFO2 = CTM_GRID( MTYPE2 ) 
XMID_4x5 = GRIDINFO2.XMID   
YMID_4x5 = GRIDINFO2.YMID   
XEDGE = GRIDINFO2.XEDGE 
YEDGE = GRIDINFO2.YEDGE

;fname = '/home/bbyrne/grid_ABoVE2.nc'
;ncdf_read,result,filename=fname,/All                                    
;lat = result.lat
;lon = result.lon

CTM_GETWEIGHT, GRIDINFO, GRIDINFO2, WEIGHT, XX_IND, YY_IND, $
             WEIGHTFILE = 'weights.025x025.to.geos5.4x5'

;[bbyrne@animus1 ~]$ ls GFED41s_025_CO
;2020  2021  2022
;[bbyrne@animus1 ~]$ pwd
;/home/bbyrne

doit=1
for yyyy=2023,2023 DO BEGIN
FOR i =0,365 DO BEGIN
   print, 'i: ', i
   CALDAT, i+366, Mnth, Dys, Ybfb, Hbfb, Mbfb, Sbfb
   fname = '/users/jk/17/bbyrne/GFED41s_025_CO/'+string(yyyy,FORMAT='(I04)')+'/'+string(Mnth,FORMAT='(I02)')+'/'+string(Dys,FORMAT='(I02)')+'.nc'
   print, fname
   
   ncdf_read,result,filename=fname,/All                                    
   lat_GFED = result.lat
   lon_GFED = result.lon
   CO_Flux_GFED = result.CO_Flux

   ;stophere

   NEP_regrid=MAKE_ARRAY(576,361,8,VALUE=0,/DOUBLE)
   NEP_regrid_ABoVE=MAKE_ARRAY(144,91,8,VALUE=0,/DOUBLE)

   if doit EQ 1 THEN BEGIN
      NEWDATA1 = CTM_REGRIDH(REFORM(CO_Flux_GFED[*,*,0]), GRIDINFO, GRIDINFO2, $
                             WFILE='weights.025x025.to.geos5.4x5' )
      doit=0
      ENDIF

   FOR indi=0,8-1 DO BEGIN
      print, indi
      ;NEP_regrid[*,*,indi] = CTM_REGRIDH(REFORM(CO_Flux_GFED[*,*,indi]), GRIDINFO, GRIDINFO2,/PER_UNIT_AREA)
   NEWDATA1 = CTM_REGRIDH(REFORM(CO_Flux_GFED[*,*,indi]), GRIDINFO, GRIDINFO2, $
                                /USE_SAVED_WEIGHTS,/PER_UNIT_AREA )
    NEP_regrid_ABoVE[*,*,indi] = NEWDATA1;[*,*,indi]
    ;stophere
   ENDFOR

; ; YMID_4x5[208:332]
; ; XMID_4x5[20:208]
; myct, 63, /rev
; WINDOW, 1
; TvMap, NEP_regrid_ABoVE[*,*,i], lon, lat, /Sample, /Coasts, /CBar, MINDATA=-3e-8, MAXDATA=3e-8,nan_color=!myct.gray     
; ; lat_GFED[439:640]
; ; lon_GFED[59:460]
; WINDOW, 2
; TvMap, CO_Flux_GFED[59:460,439:640,i], lon_GFED[59:460], lat_GFED[439:640], /Sample, /Coasts, /CBar, MINDATA=-3e-8, MAXDATA=3e-8,nan_color=!myct.gray                       

   fnameout = '/users/jk/14/bbyrne/GFED41s_CO/GFED41s_2x25_CO/'+string(yyyy,FORMAT='(I04)')+'/'+string(Mnth,FORMAT='(I02)')+'/'+string(Dys,FORMAT='(I02)')+'.nc'

   print, fnameout
   id = NCDF_CREATE(fnameout,/CLOBBER)                                            
   NCDF_ATTPUT,id,'TITLE','GFED4.1 flux 2x25 domain', /GLOBAL
   NCDF_ATTPUT,id,'Contact','Brendan Byrne (brendan.k.byrne@jpl.nasa.gov)',/GLOBAL
   NCDF_ATTPUT,id,'Date',systime(),/GLOBAL                                        
   NCDF_ATTPUT,id,'Description','CO FLUX (kgC/km2/s)', /GLOBAL
   dim0 = NCDF_DIMDEF(id, 'lon', N_ELEMENTS(XMID_4x5))
   dim1 = NCDF_DIMDEF(id, 'lat', N_ELEMENTS(YMID_4x5))
   dim2 = NCDF_DIMDEF(id, 'time', 8)            
   vid1 = NCDF_VARDEF(id, 'lon', dim0, /FLOAT)                                    
   NCDF_ATTPUT,id, vid1,'unit','degrees east'                                     
   vid2 = NCDF_VARDEF(id, 'lat', dim1, /FLOAT)                                    
   NCDF_ATTPUT,id, vid2,'unit','degrees north'                                    
   vid4 = NCDF_VARDEF(id, 'CO_Flux', [dim0,dim1,dim2], /FLOAT)                   
   NCDF_ATTPUT, id, vid4, 'long name', 'CO_Flux'                                 
   NCDF_ATTPUT,id, vid4,'unit','kgC/km2/s'                                     
   NCDF_CONTROL, id, /ENDEF                                                       
   ;NEPoutMEANarr = NEP_4x5      
   NCDF_VARPUT,id,vid1,XMID_4x5(*)                                                    
   NCDF_VARPUT,id,vid2,YMID_4x5(*)
   NCDF_VARPUT,id,vid4,NEP_regrid_ABoVE(*,*,*)                                       
   NCDF_CLOSE,id                                                                  
   CTM_CLEANUP                                                                    

ENDFOR
ENDFOR

END
